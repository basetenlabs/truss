import os
import subprocess

from helpers.context_managers import current_directory


class InferenceServerProcessController:
    def __init__(
        self,
        inference_server_home,
        inference_server_process_args,
        inference_server_port,
        app_logger,
    ) -> None:
        self._inference_server_process = None
        self._inference_server_home = inference_server_home
        self._inference_server_process_args = inference_server_process_args
        self._inference_server_port = inference_server_port
        self._inference_server_started = False
        self._app_logger = app_logger

    def start(self):
        with current_directory(self._inference_server_home):
            inf_env = os.environ.copy()
            inf_env["INFERENCE_SERVER_PORT"] = str(self._inference_server_port)
            self._inference_server_process = subprocess.Popen(
                self._inference_server_process_args,
                env=inf_env,
            )
            self._inference_server_started = True

    def stop(self):
        if self._inference_server_process is not None:
            # TODO(pankaj) send sigint wait and then kill
            poll = self._inference_server_process.poll()
            if poll is None:
                self._inference_server_process.kill()
                self._inference_server_started = False

    def inference_server_started(self) -> bool:
        return self._inference_server_started

    def is_inference_server_running(self) -> bool:
        # Explicitly check if inference server process is up, this is a bit expensive.
        if not self._inference_server_started:
            return False

        if self._inference_server_process is None:
            return False

        return self._inference_server_process.poll() is None

    def check_and_recover_inference_server(self):
        if self.inference_server_started() and not self.is_inference_server_running():
            self._app_logger.warning(
                "Inference server seems to have crashed, restarting"
            )
            self.start()
