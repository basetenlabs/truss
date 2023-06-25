import logging
import re
from pathlib import Path
from typing import Dict

from endpoints import control_app
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from helpers.errors import ModelLoadFailed, PatchApplicatonError
from helpers.inference_server_controller import InferenceServerController
from helpers.inference_server_process_controller import InferenceServerProcessController
from helpers.logging import setup_logging
from helpers.truss_patch.model_container_patch_applier import ModelContainerPatchApplier


async def handle_patch_error(_, exc):
    error_type = _camel_to_snake_case(type(exc).__name__)
    return JSONResponse(
        content={
            "error": {
                "type": error_type,
                "msg": str(exc),
            }
        }
    )


async def generic_error_handler(_, exc):
    return JSONResponse(
        content={
            "error": {
                "type": "unknown",
                "msg": f"{type(exc)}: {exc}",
            }
        }
    )


async def handle_model_load_failed(_, error):
    # Model load failures should result in 503 status
    return JSONResponse({"error": str(error)}, 503)


def create_app(base_config: Dict):
    app = FastAPI(
        title="Truss Live Reload Server",
        exception_handlers={
            PatchApplicatonError: handle_patch_error,
            ModelLoadFailed: handle_model_load_failed,
            Exception: generic_error_handler,
        },
    )

    setup_logging()

    app_logger = logging.getLogger(__name__)

    app.state.logger = app_logger

    for k, v in base_config.items():
        setattr(app.state, k, v)

    app.state.inference_server_process_controller = InferenceServerProcessController(
        app.state.inference_server_home,
        app.state.inference_server_process_args,
        app.state.inference_server_port,
        app_logger=app_logger,
    )

    pip_path = getattr(app.state, "pip_path", None)

    patch_applier = ModelContainerPatchApplier(
        Path(app.state.inference_server_home),
        app_logger,
        pip_path,
    )

    oversee_inference_server = getattr(app.state, "oversee_inference_server", True)

    app.state.inference_server_controller = InferenceServerController(
        app.state.inference_server_process_controller,
        patch_applier,
        app_logger,
        oversee_inference_server,
    )
    app.include_router(control_app)
    return app


def _camel_to_snake_case(camel_cased: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", camel_cased).lower()
