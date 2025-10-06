import asyncio
import http
import logging
import logging.config
import re
import traceback
from pathlib import Path
from typing import Awaitable, Callable, Dict

import httpx
from endpoints import control_app
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from helpers.errors import ModelLoadFailed, PatchApplicatonError
from helpers.inference_server_controller import InferenceServerController
from helpers.inference_server_process_controller import InferenceServerProcessController
from helpers.inference_server_starter import async_inference_server_startup_flow
from helpers.truss_patch.model_container_patch_applier import ModelContainerPatchApplier
from shared import log_config
from starlette.datastructures import State
from starlette.middleware.base import BaseHTTPMiddleware

SANITIZED_EXCEPTION_FRAMES = 2


# NB(nikhil): SanitizedExceptionMiddleware will reduce the noise of control server stack frames, since
# users often complain about the verbosity. Now, if any exceptions are explicitly raised during a proxied
# request, we'll log the last two stack frames which should be sufficient for debugging while significantly
# cutting down the volume.
class SanitizedExceptionMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, num_frames: int = SANITIZED_EXCEPTION_FRAMES):
        super().__init__(app)
        self.num_frames = num_frames

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        try:
            return await call_next(request)
        except Exception as exc:
            # NB(nikhil): Intentionally bypass error logging for ModelLoadFailed, since health checks
            # are noisy. The underlying model logs for why the load failed will still be visible.
            if isinstance(exc, ModelLoadFailed):
                return JSONResponse(
                    {"error": str(exc)}, status_code=http.HTTPStatus.BAD_GATEWAY.value
                )

            sanitized_traceback = self._create_sanitized_traceback(exc)
            request.app.state.logger.error(sanitized_traceback)

            if isinstance(exc, PatchApplicatonError):
                error_type = _camel_to_snake_case(type(exc).__name__)
                return JSONResponse({"error": {"type": error_type, "msg": str(exc)}})
            else:
                return JSONResponse(
                    {"error": {"type": "unknown", "msg": str(exc)}},
                    status_code=http.HTTPStatus.INTERNAL_SERVER_ERROR.value,
                )

    def _create_sanitized_traceback(self, error: Exception) -> str:
        tb_lines = traceback.format_tb(error.__traceback__)
        if tb_lines and self.num_frames > 0:
            return "".join(tb_lines[-self.num_frames :])
        return f"{type(error).__name__}: {error}"


def create_app(base_config: Dict):
    app_state = State()
    # TODO(BT-13721): better log setup: app_logger isn't captured and access log
    #   is redundant.
    logging.config.dictConfig(log_config.make_log_config("INFO"))
    app_logger = logging.getLogger(__name__)
    app_state.logger = app_logger

    for k, v in base_config.items():
        setattr(app_state, k, v)

    app_state.inference_server_process_controller = InferenceServerProcessController(
        app_state.inference_server_home,
        app_state.inference_server_process_args,
        app_state.inference_server_port,
        app_logger=app_logger,
    )

    limits = httpx.Limits(max_keepalive_connections=8, max_connections=32)
    app_state.proxy_client = httpx.AsyncClient(
        base_url=f"http://localhost:{app_state.inference_server_port}", limits=limits
    )

    uv_path = getattr(app_state, "uv_path", None)
    patch_applier = ModelContainerPatchApplier(
        Path(app_state.inference_server_home), app_logger, uv_path
    )

    oversee_inference_server = getattr(app_state, "oversee_inference_server", True)

    app_state.inference_server_controller = InferenceServerController(
        app_state.inference_server_process_controller,
        patch_applier,
        app_logger,
        oversee_inference_server,
    )

    async def start_background_inference_startup():
        asyncio.create_task(
            async_inference_server_startup_flow(
                app_state.inference_server_controller, app_logger
            )
        )

    app = FastAPI(
        title="Truss Live Reload Server",
        on_startup=[start_background_inference_startup],
    )
    app.state = app_state
    app.include_router(control_app)
    app.add_middleware(SanitizedExceptionMiddleware)

    @app.on_event("shutdown")
    def on_shutdown():
        # FastApi handles the term signal to start the shutdown flow. Here we
        # make sure that the inference server is stopped when control server
        # shuts down. Inference server has logic to wait until all requests are
        # finished before exiting. By waiting on that, we inherit the same
        # behavior for control server.
        app.state.logger.info("Term signal received, shutting down.")
        app.state.inference_server_process_controller.terminate_with_wait()

    return app


def _camel_to_snake_case(camel_cased: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", camel_cased).lower()
