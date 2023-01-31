import logging
import re
from pathlib import Path

from endpoints import control_app
from flask import Flask
from helpers.errors import PatchApplicatonError
from helpers.inference_server_controller import InferenceServerController
from helpers.inference_server_process_controller import InferenceServerProcessController
from helpers.patch_applier import PatchApplier
from werkzeug.exceptions import HTTPException


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
        app_logger=app.logger,
    )
    patch_applier = PatchApplier(
        Path(app.config["inference_server_home"]),
        app.logger,
        app.config.get("pip_path"),
    )
    app.config["inference_server_controller"] = InferenceServerController(
        app.config["inference_server_process_controller"],
        patch_applier,
        app.logger,
        app.config.get("oversee_inference_server", True),
    )
    app.register_blueprint(control_app)

    def handle_error(exc):
        try:
            raise exc
        except HTTPException:
            return exc
        except PatchApplicatonError:
            app.logger.exception(exc)
            error_type = _camel_to_snake_case(type(exc).__name__)
            return {
                "error": {
                    "type": error_type,
                    "msg": str(exc),
                }
            }
        except Exception:
            app.logger.exception(exc)
            return {
                "error": {
                    "type": "unknown",
                    "msg": f"{type(exc)}: {exc}",
                }
            }

    app.register_error_handler(Exception, handle_error)
    return app


def _camel_to_snake_case(camel_cased: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", camel_cased).lower()
