import threading
from typing import List

from helpers.inference_server_process_controller import InferenceServerProcessController
from helpers.patch_applier import PatchApplier
from helpers.types import Patch


class InferenceServerController:
    """Controls the inference server process.

    Currently, it only applies locks to various actions and mostly
    delegates to InferenceServerProcessController.
    """

    def __init__(
        self,
        process_controller: InferenceServerProcessController,
        patch_applier: PatchApplier,
    ):
        self._lock = threading.Lock()
        self._process_controller = process_controller
        self._patch_applier = patch_applier

    def apply_patch(self, patches: List[Patch]):
        with self._lock:
            self._process_controller.stop()
            for patch in patches:
                self._patch_applier.apply_patch(patch)
            self._process_controller.start()

    def restart(self):
        with self._lock:
            self._process_controller.stop()
            self._process_controller.start()

    def stop(self):
        with self._lock:
            self._process_controller.stop()
