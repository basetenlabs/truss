import asyncio
import contextlib
import http
import logging
import logging.config
import re
import traceback
from collections.abc import Awaitable
from pathlib import Path
from typing import Callable, Optional

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


# NB(nikhil): SanitizedExceptionMiddleware reduces the noise of control server
# stack frames, since users often complain about the verbosity. The headline
# line carries the request, exception type, and message, so the actual error
# is visible without a wall of frames. A small number of trailing frames from
# the full exception chain (so __cause__ is preserved) follow.
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

            # Defensive: the error handler itself must not crash. _format_error
            # touches str(exc) and the traceback chain, both of which can in
            # principle raise on pathological exceptions.
            try:
                formatted = self._format_error(request, exc)
            except Exception:
                formatted = f"Unhandled control server exception: {type(exc).__name__}"
            request.app.state.logger.error(formatted)

            if isinstance(exc, PatchApplicatonError):
                error_type = _camel_to_snake_case(type(exc).__name__)
                return JSONResponse({"error": {"type": error_type, "msg": str(exc)}})
            else:
                return JSONResponse(
                    {"error": {"type": "unknown", "msg": str(exc)}},
                    status_code=http.HTTPStatus.INTERNAL_SERVER_ERROR.value,
                )

    def _format_error(self, request: Request, error: Exception) -> str:
        lines = [
            f"{request.method} {request.url.path}: {type(error).__name__}: {error}"
        ]
        seen: set[int] = set()
        current: Optional[BaseException] = error
        is_root = True
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            if not is_root:
                lines.append(f"Caused by: {type(current).__name__}: {current}")
            lines.extend(self._format_frames(current.__traceback__))
            # Mirror Python's own rules: prefer __cause__; otherwise __context__
            # is shown unless explicitly suppressed (raise X from None).
            if current.__cause__ is not None:
                current = current.__cause__
            elif not current.__suppress_context__:
                current = current.__context__
            else:
                current = None
            is_root = False
        return "\n".join(lines)

    def _format_frames(self, tb) -> list[str]:
        if self.num_frames <= 0 or tb is None:
            return []
        # Manual formatting avoids PEP 657 caret/squiggle markers that
        # traceback.format_list adds in Python 3.11+.
        frames = traceback.extract_tb(tb)[-self.num_frames :]
        out: list[str] = []
        for f in frames:
            out.append(f'  File "{f.filename}", line {f.lineno}, in {f.name}')
        return out


def create_app(base_config: dict):
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

    @contextlib.asynccontextmanager
    async def lifespan(app):
        # Fire-and-forget: async_inference_server_startup_flow catches and logs
        # its own exceptions, so the task will never raise into asyncio's
        # "Task exception was never retrieved" handler.
        asyncio.create_task(
            async_inference_server_startup_flow(
                app_state.inference_server_controller, app_logger
            )
        )
        try:
            yield
        finally:
            # FastApi handles the term signal to start the shutdown flow. Here we
            # make sure that the inference server is stopped when control server
            # shuts down. Inference server has logic to wait until all requests are
            # finished before exiting. By waiting on that, we inherit the same
            # behavior for control server.
            app.state.logger.info("Term signal received, shutting down.")
            app.state.inference_server_process_controller.terminate_with_wait()

    app = FastAPI(title="Truss Live Reload Server", lifespan=lifespan)
    app.state = app_state
    app.include_router(control_app)
    app.add_middleware(SanitizedExceptionMiddleware)

    return app


def _camel_to_snake_case(camel_cased: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", camel_cased).lower()
