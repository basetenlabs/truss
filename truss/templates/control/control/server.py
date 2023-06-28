import asyncio
import concurrent.futures
import os
from pathlib import Path

import shared.util as utils
import uvicorn
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

    cfg = uvicorn.Config(
        application,
        host=application.state.control_server_host,
        port=application.state.control_server_port,
        workers=1,
    )
    cfg.setup_event_loop()

    max_asyncio_workers = min(32, utils.cpu_count() + 4)
    asyncio.get_event_loop().set_default_executor(
        concurrent.futures.ThreadPoolExecutor(max_workers=max_asyncio_workers)
    )

    server = uvicorn.Server(cfg)

    asyncio.run(server.serve())
