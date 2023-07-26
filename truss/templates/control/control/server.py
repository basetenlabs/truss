import asyncio
import os
from pathlib import Path

import uvicorn
from application import create_app
from helpers.inference_server_process_controller import (
    is_inference_server_intentionally_stopped,
)
from proxy import Proxy
from truss_proxy.inference_server_proxy import InferenceServerReverseProxyPlugin

PROXY_SERVER_PORT = int(os.environ.get("PROXY_SERVER_PORT", "8080"))
CONTROL_SERVER_PORT = int(os.environ.get("CONTROL_SERVER_PORT", "8091"))
INFERENCE_SERVER_PORT = int(os.environ.get("INFERENCE_SERVER_PORT", "8090"))
PYTHON_EXECUTABLE_LOOKUP_PATHS = [
    "/usr/local/bin/python",
    "/usr/local/bin/python3",
    "/usr/bin/python",
    "/usr/bin/python3",
]


def _identify_python_executable_path() -> str:
    for path in PYTHON_EXECUTABLE_LOOKUP_PATHS:
        if Path(path).exists():
            return path

    raise RuntimeError("Unable to find python, make sure it's installed.")


class ControlServer:
    def __init__(
        self,
        python_executable_path: str,
        inf_serv_home: str,
        control_server_port: int,
        inference_server_port: int,
    ):
        super().__init__()
        self._python_executable_path = python_executable_path
        self._inf_serv_home = inf_serv_home
        self._control_server_port = control_server_port
        self._inference_server_port = inference_server_port
        self._application = None

    def run(self):
        self._application = create_app(
            {
                "inference_server_home": self._inf_serv_home,
                "inference_server_process_args": [
                    "python",
                    f"{self._inf_serv_home}/inference_server.py",
                ],
                "control_server_host": "0.0.0.0",
                "control_server_port": self._control_server_port,
                "inference_server_port": self._inference_server_port,
            }
        )

        self._application.state.logger.info(
            f"Starting live reload server on port {self._control_server_port}"
        )

        cfg = uvicorn.Config(
            self._application,
            host=self._application.state.control_server_host,
            port=self._application.state.control_server_port,
        )
        cfg.setup_event_loop()

        server = uvicorn.Server(cfg)
        asyncio.run(server.serve())


if __name__ == "__main__":
    control_server = ControlServer(
        python_executable_path=_identify_python_executable_path(),
        inf_serv_home=os.environ["APP_HOME"],
        control_server_port=CONTROL_SERVER_PORT,
        inference_server_port=INFERENCE_SERVER_PORT,
    )
    InferenceServerReverseProxyPlugin.inference_server_path = (
        f"http://localhost:{INFERENCE_SERVER_PORT}/".encode()
    )
    InferenceServerReverseProxyPlugin.control_server_path = (
        f"http://localhost:{CONTROL_SERVER_PORT}/".encode()
    )
    InferenceServerReverseProxyPlugin.server_down_callback = (
        is_inference_server_intentionally_stopped
    )

    with Proxy(
        reversed=True,
        threadless=True,
        port=PROXY_SERVER_PORT,
        plugins=[InferenceServerReverseProxyPlugin],
        input_args=["--enable-reverse-proxy"],
    ):
        control_server.run()
