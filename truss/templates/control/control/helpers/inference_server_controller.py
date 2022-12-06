import os
import threading
import time

from helpers.inference_server_process_controller import InferenceServerProcessController
from helpers.patch_applier import PatchApplier
from helpers.types import Patch

INFERENCE_SERVER_CHECK_INTERVAL_SECS = 10


class InferenceServerController:
    """Controls what code the inference server runs with.

    Currently, it only applies locks to various actions and mostly
    delegates to InferenceServerProcessController.
    """

    def __init__(
        self,
        process_controller: InferenceServerProcessController,
        patch_applier: PatchApplier,
        app_logger,
        oversee_inference_server: bool = True,
    ):
        self._lock = threading.Lock()
        self._process_controller = process_controller
        self._patch_applier = patch_applier
        self._current_running_hash = os.environ.get("HASH_TRUSS", None)
        self._app_logger = app_logger
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
                raise ValueError(
                    f"Unable to apply patch: expected prev hash \
                    to be {self._current_running_hash} but found {req_prev_hash}"
                )

            patches = [
                Patch.from_dict(patch_dict) for patch_dict in patch_request["patches"]
            ]
            self._process_controller.stop()
            for patch in patches:
                self._patch_applier.apply_patch(patch)
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

    def _check_and_recover_inference_server(self):
        self._app_logger.info("Inference server overseer thread started")
        while True:
            with self._lock:
                self._process_controller.check_and_recover_inference_server()
            time.sleep(INFERENCE_SERVER_CHECK_INTERVAL_SECS)
