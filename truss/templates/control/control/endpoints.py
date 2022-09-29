import logging
import sys

from flask import Blueprint, current_app, request
from helpers.types import Patch

control_app = Blueprint("control", __name__)


logger = logging.getLogger(__name__)


@control_app.route("/patch", methods=["POST"])
def patch():
    body = request.get_json()
    try:
        patches = [Patch.from_dict(patch_dict) for patch_dict in body]
        current_app.config["inference_server_controller"].apply_patch(patches)
        logger.info("Patch applied successfully")
    except Exception:  # noqa
        ex_type, ex_value, _ = sys.exc_info()
        error_msg = f"Failed to apply patch: {ex_type}, {ex_value}"
        logger.warning(error_msg)
        return {"error": error_msg}

    return {"msg": "Patch applied successfully"}


@control_app.route("/restart_inference_server", methods=["POST"])
def restart_inference_server():
    try:
        current_app.config["inference_server_controller"].restart()
    except Exception:  # noqa
        ex_type, ex_value, _ = sys.exc_info()
        return {"error": f"Failed to restart inference server: {ex_type}, {ex_value}"}

    return {"msg": "Inference server started successfully"}


@control_app.route("/stop_inference_server", methods=["POST"])
def stop_inference_server():
    current_app.config["inference_server_controller"].stop()
    return {"msg": "Inference server stopped successfully"}
