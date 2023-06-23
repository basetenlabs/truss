import os
from pathlib import Path

from application import create_app

CONTROL_SERVER_PORT = int(os.environ.get("CONTROL_SERVER_PORT", "8080"))
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


if __name__ == "__main__":
    from shared.uvicorn_config import start_uvicorn_server

    inf_serv_home: str = os.environ["APP_HOME"]
    python_executable_path: str = _identify_python_executable_path()
    application = create_app(
        {
            "inference_server_home": inf_serv_home,
            "inference_server_process_args": [
                python_executable_path,
                f"{inf_serv_home}/inference_server.py",
            ],
            "control_server_host": "0.0.0.0",
            "control_server_port": CONTROL_SERVER_PORT,
            "inference_server_port": INFERENCE_SERVER_PORT,
        }
    )

    application.state.logger.info(
        f"Starting live reload server on port {CONTROL_SERVER_PORT}"
    )

    start_uvicorn_server(
        application,
        host=application.state.control_server_host,
        port=application.state.control_server_port,
    )
