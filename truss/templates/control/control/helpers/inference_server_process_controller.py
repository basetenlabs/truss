import os
import subprocess

from helpers.context_managers import current_directory


class InferenceServerProcessController:
    def __init__(
        self,
        inference_server_home,
        inference_server_process_args,
        inference_server_port,
    ) -> None:
        self._inference_server_process = None
        self._inference_server_home = inference_server_home
        self._inference_server_process_args = inference_server_process_args
        self._inference_server_port = inference_server_port
        self._inference_server_running = False

    def start(self):
        with current_directory(self._inference_server_home):
            inf_env = os.environ.copy()
            inf_env["INFERENCE_SERVER_PORT"] = str(self._inference_server_port)
            self._inference_server_process = subprocess.Popen(
                self._inference_server_process_args,
                env=inf_env,
            )
            self._inference_server_running = True

    def stop(self):
        if self._inference_server_process is not None:
            # TODO(pankaj) send sigint wait and then kill
            poll = self._inference_server_process.poll()
            if poll is None:
                self._inference_server_process.kill()
                self._inference_server_running = False

    def inference_server_running(self) -> bool:
        return self._inference_server_running
