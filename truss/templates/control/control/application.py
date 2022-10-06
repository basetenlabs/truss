import logging
from pathlib import Path

from endpoints import control_app
from flask import Flask
from helpers.inference_server_controller import InferenceServerController
from helpers.inference_server_process_controller import InferenceServerProcessController
from helpers.patch_applier import PatchApplier


def create_app(base_config: dict):
    app = Flask(__name__)
    # TODO(pankaj): change this back to info once things are stable
    app.logger.setLevel(logging.DEBUG)
    app.config.update(base_config)
    app.config[
        "inference_server_process_controller"
    ] = InferenceServerProcessController(
        app.config["inference_server_home"],
        app.config["inference_server_process_args"],
        app.config["inference_server_port"],
    )
    patch_applier = PatchApplier(Path(app.config["inference_server_home"]), app.logger)
    app.config["inference_server_controller"] = InferenceServerController(
        app.config["inference_server_process_controller"], patch_applier, app.logger
    )
    app.register_blueprint(control_app)
    return app
