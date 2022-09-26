import os

import requests
from application import create_app

DEFAULT_CONTROL_SERVER_PORT = 8090


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
            "control_server_port": DEFAULT_CONTROL_SERVER_PORT,
        }
    )

    # Startup the inference server
    # todo: move to constant
    patch_ping_url = os.environ.get("PATCH_PING_URL_TRUSS", None)
    if patch_ping_url is None:
        application.config["inference_server_controller"].restart()
    else:
        # In this flow the other party needs to call patch, which would start
        # the inference server.
        # todo: add retries here
        requests.post(patch_ping_url)

    print(f"Starting control server on port {DEFAULT_CONTROL_SERVER_PORT}")
    server = create_server(
        application,
        host=application.config["control_server_host"],
        port=application.config["control_server_port"],
    )
    server.run()
