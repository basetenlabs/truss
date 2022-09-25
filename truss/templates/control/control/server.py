import os
import sys
from typing import Callable

from flask import Flask, request
from helpers.inference_server_controller import InferenceServerController
from helpers.inference_server_process_controller import InferenceServerProcessController
from helpers.types import Patch

app = Flask(__name__)


DEFAULT_CONTROL_SERVER_PORT = 8090


@app.route("/patch", methods=["POST"])
def patch():
    body = request.get_json()
    patch = Patch.from_dict(body)
    try:
        app.config["inference_server_controller"].apply_patch(patch)
    except Exception:  # noqa
        ex_type, ex_value, _ = sys.exc_info()
        return {"error": f"Failed to apply patch: {ex_type}, {ex_value}"}

    return {"msg": "Patch applied successfully"}


@app.route("/restart_inference_server", methods=["POST"])
def restart_inference_server():
    try:
        app.config["inference_server_controller"].restart()
    except Exception:  # noqa
        ex_type, ex_value, _ = sys.exc_info()
        return {"error": f"Failed to restart inference server: {ex_type}, {ex_value}"}

    return {"msg": "Inference server started successfully"}


@app.route("/stop_inference_server")
def stop_inference_server():
    app.config["inference_server_controller"].stop()
    return {"msg": "Inference server stopped successfully"}


def _noop(*args, **kwargs):
    pass


def _make_server(configure: Callable = _noop):
    from waitress import create_server

    configure(app)
    app.config[
        "inference_server_process_controller"
    ] = InferenceServerProcessController(
        app.config["inference_server_home"], app.config["inference_server_process_args"]
    )
    app.config["inference_server_controller"] = InferenceServerController(
        app.config["inference_server_process_controller"],
    )

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
