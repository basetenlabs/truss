import logging
import re
from pathlib import Path
from typing import Dict

from endpoints import control_app
from fastapi import FastAPI

# from helpers.errors import PatchApplicatonError
from helpers.inference_server_controller import InferenceServerController
from helpers.inference_server_process_controller import InferenceServerProcessController
from helpers.patch_applier import PatchApplier


def create_app(base_config: Dict):
    app = FastAPI(title="Truss Live Reload Server")

    app_logger = logging.getLogger(__name__)
    # TODO(pankaj): change this back to info once things are stable
    app_logger.setLevel(logging.DEBUG)

    app.state.logger = app_logger

    for k, v in base_config.items():
        setattr(app.state, k, v)

    app.state.inference_server_process_controller = InferenceServerProcessController(
        app.state.inference_server_home,
        app.state.inference_server_process_args,
        app.state.inference_server_port,
        app_logger=app_logger,
    )
    pip_path = None
    try:
        pip_path = app.state.pip_path
    except AttributeError:
        pass

    patch_applier = PatchApplier(
        Path(app.state.inference_server_home),
        app_logger,
        pip_path,
    )
    oversee_inference_server = True
    try:
        oversee_inference_server = app.state.oversee_inference_server
    except AttributeError:
        pass

    app.state.inference_server_controller = InferenceServerController(
        app.state.inference_server_process_controller,
        patch_applier,
        app_logger,
        oversee_inference_server,
    )
    app.include_router(control_app)

    # TODO: use proper FastAPI exception handling here
    # def handle_error(exc):
    #     try:
    #         raise exc
    #     except HTTPException:
    #         return exc
    #     except PatchApplicatonError:
    #         app.logger.exception(exc)
    #         error_type = _camel_to_snake_case(type(exc).__name__)
    #         return {
    #             "error": {
    #                 "type": error_type,
    #                 "msg": str(exc),
    #             }
    #         }
    #     except Exception:
    #         app.logger.exception(exc)
    #         return {
    #             "error": {
    #                 "type": "unknown",
    #                 "msg": f"{type(exc)}: {exc}",
    #             }
    #         }

    # app.register_error_handler(Exception, handle_error)
    return app


def _camel_to_snake_case(camel_cased: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", camel_cased).lower()
