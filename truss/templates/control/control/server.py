import os
import subprocess
import sys
import threading
from typing import Callable

from control_utils.context_managers import current_directory
from flask import Flask

app = Flask(__name__)


DEFAULT_CONTROL_SERVER_PORT = 8090
INFERENCE_SERVER_PROCESS = None
RESTART_INFERENCE_SERVER_LOCK = threading.Lock()


@app.route("/patch", methods=["POST"])
def patch():
    return "<p>Hello, World!</p>"


@app.route("/restart_inference_server", methods=["POST"])
def restart_inference_server():
    global INFERENCE_SERVER_PROCESS
    global RESTART_INFERENCE_SERVER_LOCK

    with RESTART_INFERENCE_SERVER_LOCK:
        try:
            if INFERENCE_SERVER_PROCESS is not None:
                # TODO(pankaj) send sigint wait and then kill
                INFERENCE_SERVER_PROCESS.kill()

            with current_directory(app.config["inference_server_home"]):
                INFERENCE_SERVER_PROCESS = subprocess.Popen(
                    app.config["inference_server_process_args"]
                )
        except Exception:  # noqa
            ex_type, ex_value, _ = sys.exc_info()
            return {
                "error": f"Failed to restart inference server: {ex_type}, {ex_value}"
            }

    return {"msg": "Inference server started successfully"}


@app.route("/stop_inference_server")
def stop_inference_server():
    # todo
    return {"msg": "Inference server stopped successfully"}


def _noop(*args, **kwargs):
    pass


def _make_server(configure: Callable = _noop):
    from waitress import create_server

    configure(app)
    print(f"Starting control server on port {DEFAULT_CONTROL_SERVER_PORT}")
    return create_server(
        app,
        host=app.config["control_server_host"],
        port=app.config["control_server_port"],
    )


if __name__ == "__main__":

    def configure_app(flask_app):
        flask_app.config["inference_server_home"] = os.environ["APP_HOME"]
        flask_app.config["inference_server_process_args"] = [
            "/usr/local/bin/python",
            f"{flask_app.config['inference_server_home']}/inference_server.py",
        ]
        flask_app.config["control_server_host"] = "0.0.0.0"
        flask_app.config["control_server_port"] = DEFAULT_CONTROL_SERVER_PORT

    server = _make_server(configure_app)
    server.run()
