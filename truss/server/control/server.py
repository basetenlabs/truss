import asyncio
import os
from pathlib import Path

import uvicorn
from truss.server.control.application import create_app

CONTROL_SERVER_PORT = int(os.environ.get("CONTROL_SERVER_PORT", "8080"))
INFERENCE_SERVER_PORT = int(os.environ.get("INFERENCE_SERVER_PORT", "8090"))


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

    def run(self):
        application = create_app(
            {
                "inference_server_home": self._inf_serv_home,
                "inference_server_process_args": [
                    self._python_executable_path,
                    "-m",
                    "truss.server.inference_server",
                ],
                "control_server_host": "0.0.0.0",
                "control_server_port": self._control_server_port,
                "inference_server_port": self._inference_server_port,
            }
        )

        application.state.logger.info(
            f"Starting live reload server on port {self._control_server_port}"
        )

        cfg = uvicorn.Config(
            application,
            host=application.state.control_server_host,
            port=application.state.control_server_port,
            # We hard-code the http parser as h11 (the default) in case the user has
            # httptools installed, which does not work with our requests & version
            # of uvicorn.
            http="h11",
        )
        cfg.setup_event_loop()

        server = uvicorn.Server(cfg)
        asyncio.run(server.serve())


if __name__ == "__main__":
    control_server = ControlServer(
        python_executable_path=os.environ.get("PYTHON_EXECUTABLE", default="python3"),
        inf_serv_home=os.environ.get("APP_HOME", default=str(Path.cwd())),
        control_server_port=CONTROL_SERVER_PORT,
        inference_server_port=INFERENCE_SERVER_PORT,
    )
    control_server.run()
