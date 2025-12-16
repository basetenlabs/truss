import logging
import subprocess
import time
from pathlib import Path
from typing import Optional

from helpers.context_managers import current_directory
from shared.util import kill_child_processes

INFERENCE_SERVER_FAILED_FILE = Path("~/inference_server_crashed.txt").expanduser()
TERMINATION_TIMEOUT_SECS = 120.0
TERMINATION_CHECK_INTERVAL_SECS = 0.5


class InferenceServerProcessController:
    _inference_server_process: Optional[subprocess.Popen] = None
    _inference_server_port: int
    _inference_server_home: str
    _app_logger: logging.Logger
    _inference_server_process_args: list[str]
    _logged_unrecoverable_since_last_restart: bool

    def __init__(
        self,
        inference_server_home: str,
        inference_server_process_args: list[str],
        inference_server_port: int,
        app_logger: logging.Logger,
    ) -> None:
        self._inference_server_home = inference_server_home
        self._inference_server_process_args = inference_server_process_args
        self._inference_server_port = inference_server_port
        self._inference_server_started = False
        self._inference_server_ever_started = False
        self._inference_server_terminated = False
        self._logged_unrecoverable_since_last_restart = False
        self._app_logger = app_logger

    def start(self, inf_env: dict):
        with current_directory(self._inference_server_home):
            inf_env["INFERENCE_SERVER_PORT"] = str(self._inference_server_port)
            self._inference_server_process = subprocess.Popen(
                self._inference_server_process_args, env=inf_env
            )

            self._inference_server_started = True
            self._inference_server_ever_started = True
            self._logged_unrecoverable_since_last_restart = False

    def _terminate_children_and_process(self):
        """Kill child processes first, then parent. Prevents port binding conflicts."""
        # Use a shorter timeout than the truss patch read timeout (=120s):
        # see remote/baseten/api.py:_post_graphql_query()
        kill_child_processes(self._inference_server_process.pid, timeout_seconds=30)
        self._inference_server_process.terminate()

    def stop(self):
        if self._inference_server_process is not None:
            self._terminate_children_and_process()
            self._inference_server_process.wait()

        self._inference_server_started = False

    def terminate_with_wait(self):
        self._terminate_children_and_process()
        self._inference_server_terminated = True
        termination_check_attempts = int(
            TERMINATION_TIMEOUT_SECS / TERMINATION_CHECK_INTERVAL_SECS
        )
        for _ in range(termination_check_attempts):
            time.sleep(TERMINATION_CHECK_INTERVAL_SECS)
            # None returncode means alive
            # https://docs.python.org/3.9/library/subprocess.html#subprocess.Popen.returncode
            if self._inference_server_process.poll() is not None:
                return

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

    def is_inference_server_intentionally_stopped(self) -> bool:
        return INFERENCE_SERVER_FAILED_FILE.exists()

    def is_inference_server_terminated(self) -> bool:
        return self._inference_server_terminated

    def check_and_recover_inference_server(self, inf_env: dict):
        if (
            self.inference_server_started()
            and not self.is_inference_server_running()
            and not self.is_inference_server_terminated()
        ):
            if not self.is_inference_server_intentionally_stopped():
                self._app_logger.warning(
                    "Inference server seems to have crashed, restarting"
                )
                self.start(inf_env)
            else:
                if not self._logged_unrecoverable_since_last_restart:
                    self._app_logger.warning(
                        "Inference server unrecoverable. Try patching"
                    )
                    self._logged_unrecoverable_since_last_restart = True
