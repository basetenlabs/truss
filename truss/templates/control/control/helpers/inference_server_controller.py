import logging
import os
import threading
import time

from helpers.errors import (
    InadmissiblePatch,
    PatchFailedRecoverable,
    PatchFailedUnrecoverable,
    UnsupportedPatch,
)
from helpers.inference_server_process_controller import InferenceServerProcessController
from helpers.patch_applier import PatchApplier
from helpers.types import Patch, PatchType

INFERENCE_SERVER_CHECK_INTERVAL_SECS = 10


class InferenceServerController:
    """Controls what code the inference server runs with.

    Currently, it only applies locks to various actions and mostly
    delegates to InferenceServerProcessController.
    """

    _process_controller: InferenceServerProcessController
    _patch_applier: PatchApplier
    _app_logger: logging.Logger
    _oversee_inference_server: bool

    def __init__(
        self,
        process_controller: InferenceServerProcessController,
        patch_applier: PatchApplier,
        app_logger: logging.Logger,
        oversee_inference_server: bool = True,
    ):
        self._lock = threading.Lock()
        self._process_controller = process_controller
        self._patch_applier = patch_applier
        self._current_running_hash = os.environ.get("HASH_TRUSS", None)
        self._app_logger = app_logger
        self._has_partially_applied_patch = False
        if oversee_inference_server:
            self._inference_server_overseer_thread = threading.Thread(
                target=self._check_and_recover_inference_server
            )
            self._inference_server_overseer_thread.start()

    def apply_patch(self, patch_request):
        with self._lock:
            req_hash = patch_request["hash"]
            req_prev_hash = patch_request["prev_hash"]

            self._app_logger.debug(
                f"request hash: {req_hash}, request prev hash:"
                f" {req_prev_hash}, current hash: "
                f"{self._current_running_hash}"
            )

            if req_hash == self._current_running_hash:
                # We are in sync, ok to start running inference server now, if
                # it's not running.
                if not self._process_controller.inference_server_started():
                    self._process_controller.start()
                self._app_logger.info("Request hash same as current hash, skipping.")
                return

            if req_prev_hash != self._current_running_hash:
                raise InadmissiblePatch(
                    f"Unable to apply patch: expected prev hash \
                    to be {self._current_running_hash} but found {req_prev_hash}"
                )

            try:
                patches = [
                    Patch.from_dict(patch_dict)
                    for patch_dict in patch_request["patches"]
                ]
            except (KeyError, ValueError) as exc:
                raise UnsupportedPatch(str(exc)) from exc

            self._process_controller.stop()
            patches.sort(key=_patch_sort_key_fn)
            try:
                patches_executed = 0
                for patch in patches:
                    self._patch_applier.apply_patch(patch)
                    patches_executed += 1
            except Exception as exc:
                if patches_executed > 0:
                    # In this case we leave inference server stopped, to reflect
                    # the bad state; with partially applied patch all bets are off.
                    # Correct handling of this scenario is to fallback to full deploy.
                    self._has_partially_applied_patch = True
                    raise PatchFailedUnrecoverable(str(exc)) from exc
                # No patches executed, the very first patch failed. We
                # consider this safe to start inference server back up.
                # Theoretically, a single patch application may leave
                # side-effects, but that's not the case with currently
                # supported patches.
                self._process_controller.start()
                raise PatchFailedRecoverable(str(exc)) from exc

            self._process_controller.start()
            self._current_running_hash = req_hash

    def truss_hash(self) -> str:
        return self._current_running_hash

    def restart(self):
        with self._lock:
            self._process_controller.stop()
            self._process_controller.start()

    def start(self):
        # For now, start just does restart, this alias allows for better
        # readability in certain scenarios where the intention is to start.
        self.restart()

    def stop(self):
        with self._lock:
            self._process_controller.stop()

    def has_partially_applied_patch(self):
        with self._lock:
            return self._has_partially_applied_patch

    def _check_and_recover_inference_server(self):
        self._app_logger.info("Inference server overseer thread started")
        while True:
            with self._lock:
                self._process_controller.check_and_recover_inference_server()
            time.sleep(INFERENCE_SERVER_CHECK_INTERVAL_SECS)


def _patch_sort_key_fn(patch: Patch) -> int:
    # System packages need to be applied before python requirements as they
    # might need them.
    # System packages or python packages are more likely to fail to apply then
    # model code patches, so it's better to apply model code patches last to
    # avoid ending up with partially applied patches which currently triggers
    # full deploy.
    if patch.type == PatchType.SYSTEM_PACKAGE:
        return 0

    if patch.type == PatchType.PYTHON_REQUIREMENT:
        return 1

    if patch.type == PatchType.MODEL_CODE:
        return 2

    raise ValueError(f"Unexpected patch type {patch.type}")
