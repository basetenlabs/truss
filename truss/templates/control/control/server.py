import os

import requests
from application import create_app
from tenacity import Retrying, stop_after_attempt, wait_fixed

CONTROL_SERVER_PORT = int(os.environ.get("CONTROL_SERVER_PORT", "8080"))
INFERENCE_SERVER_PORT = int(os.environ.get("INFERENCE_SERVER_PORT", "8090"))


if __name__ == "__main__":
    from waitress import create_server

    inf_serv_home = os.environ["APP_HOME"]
    application = create_app(
        {
            "inference_server_home": inf_serv_home,
            "inference_server_process_args": [
                "/usr/local/bin/python",
                f"{inf_serv_home}/inference_server.py",
            ],
            "control_server_host": "0.0.0.0",
            "control_server_port": CONTROL_SERVER_PORT,
            "inference_server_port": INFERENCE_SERVER_PORT,
        }
    )

    patch_ping_url = os.environ.get("PATCH_PING_URL_TRUSS", None)
    if patch_ping_url is None:
        application.config["inference_server_controller"].restart()
    else:
        # In this flow the other party needs to call patch, which would start
        # the inference server.
        for attempt in Retrying(
            stop=stop_after_attempt(3),
            wait=wait_fixed(1),
        ):
            with attempt:
                # Fire and forget
                try:
                    application.logger.info(f"Pinging {patch_ping_url} for patch")
                    requests.post(patch_ping_url, timeout=1)
                except requests.Timeout:
                    pass

    application.logger.info(f"Starting control server on port {CONTROL_SERVER_PORT}")
    server = create_server(
        application,
        host=application.config["control_server_host"],
        port=application.config["control_server_port"],
    )
    server.run()
