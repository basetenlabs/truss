import logging
import os
import signal
import subprocess

from helpers.context_managers import current_directory
from typing import List


class InferenceServerProcessController:

    _inference_server_process: subprocess.Popen = None
    _inference_server_port: int
    _inference_server_home: str
    _app_logger: logging.Logger
    _inference_server_process_args: List[str]

    def __init__(
        self,
        inference_server_home: str,
        inference_server_process_args: List[str],
        inference_server_port: int,
        app_logger: logging.Logger,
    ) -> None:
        self._inference_server_home = inference_server_home
        self._inference_server_process_args = inference_server_process_args
        self._inference_server_port = inference_server_port
        self._inference_server_started = False
        self._inference_server_ever_started = False
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
            self._inference_server_ever_started = True

    def stop(self):
        if self._inference_server_process is not None:
            name = " ".join(self._inference_server_process_args)

            for line in os.popen("ps ax | grep '" + name + "' | grep -v grep"):
                pid = line.split()[0]
                os.kill(int(pid), signal.SIGKILL)

        self._inference_server_started = False

    def inference_server_started(self) -> bool:
        return self._inference_server_started

    def inference_server_ever_started(self) -> bool:
        return self._inference_server_ever_started

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
