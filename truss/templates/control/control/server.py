import os
from threading import Thread

from application import create_app
from helpers.inference_server_starter import inference_server_startup_flow

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

    # Perform inference server startup flow in background
    Thread(target=inference_server_startup_flow, args=(application,)).start()

    application.logger.info(f"Starting control server on port {CONTROL_SERVER_PORT}")
    server = create_server(
        application,
        host=application.config["control_server_host"],
        port=application.config["control_server_port"],
    )
    server.run()
