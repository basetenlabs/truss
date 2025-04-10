import asyncio
import json
import logging
import logging.config
import os
import signal
import sys
from http import HTTPStatus
from pathlib import Path
from typing import TYPE_CHECKING, Awaitable, Callable, Dict, Optional, Union

import pydantic
import uvicorn
import yaml
from common import errors, tracing
from common.schema import TrussSchema
from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import ORJSONResponse, StreamingResponse
from fastapi.routing import APIRoute as FastAPIRoute
from fastapi.routing import APIWebSocketRoute as FastAPIWebSocketRoute
from model_wrapper import ModelWrapper
from opentelemetry import propagate as otel_propagate
from opentelemetry import trace
from opentelemetry.sdk import trace as sdk_trace
from prometheus_client import (
    REGISTRY,
    gc_collector,
    make_asgi_app,
    metrics,
    platform_collector,
    process_collector,
)
from pydantic import BaseModel
from shared import log_config, serialization
from shared.secrets_resolver import SecretsResolver
from starlette.requests import ClientDisconnect
from starlette.responses import Response

if sys.version_info >= (3, 9):
    from typing import AsyncGenerator, Generator
else:
    from typing_extensions import AsyncGenerator, Generator

PYDANTIC_MAJOR_VERSION = int(pydantic.VERSION.split(".")[0])

# [IMPORTANT] A lot of things depend on this currently, change with extreme care.
TIMEOUT_GRACEFUL_SHUTDOWN = 120
INFERENCE_SERVER_FAILED_FILE = Path("~/inference_server_crashed.txt").expanduser()

# Hardcoded 100MiB message maximum on websocket connections.
# TODO(bryanzhang) Align this with other websocket components so it's not so
# difficult to change.
WS_MAX_MSG_SZ_BYTES = 100 * (1 << 20)

if TYPE_CHECKING:
    from model_wrapper import InputType, OutputType


async def parse_body(request: Request) -> bytes:
    """
    Used by FastAPI to read body in an asynchronous manner
    """
    try:
        return await request.body()
    except ClientDisconnect as exc:
        error_message = "Client disconnected while reading request."
        logging.warning(error_message)
        raise HTTPException(status_code=499, detail=error_message) from exc


async def _safe_close_websocket(
    ws: WebSocket, reason: Optional[str], status_code: int = 1000
) -> None:
    try:
        await ws.close(code=status_code, reason=reason)
    except RuntimeError as close_error:
        logging.debug(f"Duplicate close of websocket: `{close_error}`.")


class BasetenEndpoints:
    """The implementation of the model server endpoints.

    Historically, we relied on the kserve server interface, which assumes that
    multiple models are running behind a registry. As a result, some arguments to
    to functions will rename unused except for backwards compatibility checks.
    """

    def __init__(self, model: ModelWrapper, tracer: sdk_trace.Tracer) -> None:
        self._model = model
        self._tracer = tracer

    def check_healthy(self):
        if self._model.load_failed:
            INFERENCE_SERVER_FAILED_FILE.touch()
            os.kill(os.getpid(), signal.SIGKILL)

        if not self._model.ready:
            raise errors.ModelNotReady(self._model.name)

    async def model_ready(self, model_name: str) -> dict:
        is_healthy = await self._model.is_healthy()
        if is_healthy is None:
            self.check_healthy()
        elif not is_healthy:
            raise errors.ModelNotReady(self._model.name)

        return {}

    async def model_loaded(self, model_name: str) -> dict:
        self.check_healthy()
        return {}

    async def invocations_ready(self) -> Dict[str, Union[str, bool]]:
        """
        This method provides compatibility with Sagemaker hosting for the 'ping' endpoint.
        """
        if self._model is None:
            raise errors.ModelMissingError("model")
        self.check_healthy()

        return {}

    async def invocations(
        self, request: Request, body_raw: bytes = Depends(parse_body)
    ) -> Response:
        """
        This method provides compatibility with Sagemaker hosting for the 'invocations' endpoint.
        """
        return await self.predict(self._model.name, request, body_raw)

    async def _parse_body(
        self,
        request: Request,
        body_raw: bytes,
        truss_schema: Optional[TrussSchema],
        span: trace.Span,
    ) -> "InputType":
        if self.is_binary(request):
            with tracing.section_as_event(span, "binary-deserialize"):
                inputs = serialization.truss_msgpack_deserialize(body_raw)
            if truss_schema:
                try:
                    with tracing.section_as_event(span, "parse-pydantic"):
                        inputs = truss_schema.input_type.parse_obj(inputs)
                except pydantic.ValidationError as e:
                    raise errors.InputParsingError(
                        errors.format_pydantic_validation_error(e)
                    ) from e
        else:
            if truss_schema:
                try:
                    with tracing.section_as_event(span, "parse-pydantic"):
                        inputs = truss_schema.input_type.parse_raw(body_raw)
                except pydantic.ValidationError as e:
                    raise errors.InputParsingError(
                        errors.format_pydantic_validation_error(e)
                    ) from e
            else:
                try:
                    with tracing.section_as_event(span, "json-deserialize"):
                        inputs = json.loads(body_raw)
                except json.JSONDecodeError as e:
                    raise errors.InputParsingError(
                        f"Invalid JSON payload: {str(e)}"
                    ) from e

        return inputs

    async def _execute_request(
        self,
        method: Callable[["InputType", Request], Awaitable["OutputType"]],
        request: Request,
        body_raw: bytes,
    ) -> Response:
        """
        Executes a predictive endpoint
        """
        self.check_healthy()
        trace_ctx = otel_propagate.extract(request.headers) or None
        # This is the top-level span in the truss-server, so we set the context here.
        # Nested spans "inherit" context automatically.
        with self._tracer.start_as_current_span(
            f"{method.__name__}-endpoint", context=trace_ctx
        ) as span:
            inputs: Optional["InputType"]
            if self._model.skip_input_parsing:
                inputs = None
            else:
                inputs = await self._parse_body(
                    request, body_raw, self._model.truss_schema, span
                )
            with tracing.section_as_event(span, "model-call"):
                result: "OutputType" = await method(inputs, request)

            # In the case that the model returns a Generator object, return a
            # StreamingResponse instead.
            if isinstance(result, (AsyncGenerator, Generator)):
                # media_type in StreamingResponse sets the Content-Type header
                return StreamingResponse(result, media_type="application/octet-stream")
            elif isinstance(result, Response):
                if result.status_code >= HTTPStatus.MULTIPLE_CHOICES.value:
                    errors.add_error_headers_to_user_response(result)
                return result
            return self._serialize_result(result, self.is_binary(request), span)

    async def predict(
        self, model_name: str, request: Request, body_raw: bytes = Depends(parse_body)
    ) -> Response:
        return await self._execute_request(
            method=self._model.predict, request=request, body_raw=body_raw
        )

    async def chat_completions(
        self, request: Request, body_raw: bytes = Depends(parse_body)
    ) -> Response:
        return await self._execute_request(
            method=self._model.chat_completions, request=request, body_raw=body_raw
        )

    async def completions(
        self, request: Request, body_raw: bytes = Depends(parse_body)
    ) -> Response:
        return await self._execute_request(
            method=self._model.completions, request=request, body_raw=body_raw
        )

    async def websocket(self, ws: WebSocket) -> None:
        self.check_healthy()
        trace_ctx = otel_propagate.extract(ws.headers) or None
        # We don't go through the typical execute_request path, since we don't need
        # to parse request body or attempt to serialize results.
        with self._tracer.start_as_current_span("websocket", context=trace_ctx):
            if not self._model.model_descriptor.websocket:
                msg = "WebSocket is not implemented on this deployment."
                logging.error(msg)
                # Ideally we would send a response before accepting the WS, but it is
                # hard to customize the denied upgrade request, so
                # instead we go the clumsy way of sending the error response through
                # accepted WS itself.
                try:
                    await ws.accept()
                    await ws.close(code=1003, reason=msg)
                    return
                except WebSocketDisconnect:
                    return

            with errors.intercept_exceptions(
                logging.getLogger(), self._model.model_file_name
            ):
                try:
                    await ws.accept()
                    await self._model.websocket(ws)
                    await _safe_close_websocket(ws, None, status_code=1000)
                except WebSocketDisconnect as ws_error:
                    logging.info(
                        f"Client terminated websocket connection: `{ws_error}`."
                    )
                except Exception:
                    await _safe_close_websocket(
                        ws, errors.MODEL_ERROR_MESSAGE, status_code=1011
                    )
                    raise  # Re raise to let `intercept_exceptions` deal with it.

    def _serialize_result(
        self, result: "OutputType", is_binary: bool, span: trace.Span
    ) -> Response:
        response_headers = {}
        if is_binary:
            if isinstance(result, BaseModel):
                with tracing.section_as_event(span, "binary-dump"):
                    if PYDANTIC_MAJOR_VERSION > 1:
                        result = result.model_dump(mode="python")
                    else:
                        result = result.dict()
            # If the result is not already serialize and not a pydantic model, it must
            # be something that can be serialized with `truss_msgpack_serialize` (some
            # dict / nested structure).
            if not isinstance(result, bytes):
                with tracing.section_as_event(span, "binary-serialize"):
                    result = serialization.truss_msgpack_serialize(result)

            response_headers["Content-Type"] = "application/octet-stream"
            return Response(content=result, headers=response_headers)
        else:
            with tracing.section_as_event(span, "json-serialize"):
                if isinstance(result, BaseModel):
                    # Note: chains has a pydantic integration for numpy arrays
                    # `NumpyArrayField`. `result.dict()`, passes through the array
                    # object which cannot be JSON serialized.
                    # In pydantic v2 `result.model_dump(mode="json")` could be used.
                    # For backwards compatibility we dump directly the JSON string.
                    content = result.json()
                else:
                    content = json.dumps(result, cls=serialization.DeepNumpyEncoder)

                response_headers["Content-Type"] = "application/json"
                return Response(content=content, headers=response_headers)

    async def schema(self, model_name: str) -> Dict:
        if self._model.truss_schema is None:
            # If there is not a TrussSchema, we return a 404.
            if self._model.ready:
                raise HTTPException(status_code=404, detail="No schema found")
            else:
                raise HTTPException(
                    status_code=503,
                    detail="Schema not available, please try again later.",
                )
        else:
            return self._model.truss_schema.serialize()

    @staticmethod
    def is_binary(request: Request):
        return (
            "Content-Type" in request.headers
            and request.headers["Content-Type"] == "application/octet-stream"
        )


class TrussServer:
    """This wrapper class manages creation and cleanup of uvicorn server processes
    running the FastAPI inference server app.

    TrussServer runs as a main process managing UvicornCustomServer subprocesses that
    in turn may manage their own worker processes. Notably, this main process is kept
    alive when running `servers_task()` because of the child uvicorn server processes'
    main loop.
    """

    _server: Optional[uvicorn.Server]

    def __init__(self, http_port: int, config_or_path: Union[str, Path, Dict]):
        # This is run before uvicorn is up. Need explicit logging config here.
        logging.config.dictConfig(log_config.make_log_config("INFO"))

        if isinstance(config_or_path, (str, Path)):
            with open(config_or_path, encoding="utf-8") as config_file:
                config = yaml.safe_load(config_file)
        else:
            config = config_or_path

        secrets = SecretsResolver.get_secrets(config)
        tracer = tracing.get_truss_tracer(secrets, config)
        self._http_port = http_port
        self._config = config
        self._model = ModelWrapper(self._config, tracer)
        self._endpoints = BasetenEndpoints(self._model, tracer)
        self._server = None

    def cleanup(self):
        if INFERENCE_SERVER_FAILED_FILE.exists():
            INFERENCE_SERVER_FAILED_FILE.unlink()

    def on_startup(self):
        """
        This method will be started inside the main process, so here is where
        we want to setup our logging and model.
        """
        self.cleanup()
        self._model.start_load_thread()
        asyncio.create_task(self._shutdown_if_load_fails())
        self._model.setup_polling_for_environment_updates()

    async def _shutdown_if_load_fails(self):
        while not self._model.ready:
            await asyncio.sleep(0.5)
            if self._model.load_failed:
                assert self._server is not None
                logging.info("Trying shut down after failed model load.")
                self._server.should_exit = True
                return

    def create_application(self):
        app = FastAPI(
            title="Baseten Inference Server",
            docs_url=None,
            redoc_url=None,
            default_response_class=ORJSONResponse,
            on_startup=[self.on_startup],
            routes=[
                # liveness endpoint
                FastAPIRoute(r"/", lambda: True),
                # readiness endpoint
                FastAPIRoute(
                    r"/v1/models/{model_name}", self._endpoints.model_ready, tags=["V1"]
                ),
                # loaded endpoint
                FastAPIRoute(
                    r"/v1/models/{model_name}/loaded",
                    self._endpoints.model_loaded,
                    tags=["V1"],
                ),
                FastAPIRoute(
                    r"/v1/models/{model_name}/schema",
                    self._endpoints.schema,
                    methods=["GET"],
                    tags=["V1"],
                ),
                FastAPIRoute(
                    r"/v1/models/{model_name}:predict",
                    self._endpoints.predict,
                    methods=["POST"],
                    tags=["V1"],
                ),
                FastAPIRoute(
                    r"/v1/models/{model_name}:predict_binary",
                    self._endpoints.predict,
                    methods=["POST"],
                    tags=["V1"],
                ),
                # OpenAI Spec
                FastAPIRoute(
                    r"/v1/chat/completions",
                    self._endpoints.chat_completions,
                    methods=["POST"],
                    tags=["V1"],
                ),
                FastAPIRoute(
                    r"/v1/completions",
                    self._endpoints.completions,
                    methods=["POST"],
                    tags=["V1"],
                ),
                # Websocket endpoint
                FastAPIWebSocketRoute(r"/v1/websocket", self._endpoints.websocket),
                # Endpoint aliases for Sagemaker hosting
                FastAPIRoute(r"/ping", self._endpoints.invocations_ready),
                FastAPIRoute(
                    r"/invocations", self._endpoints.invocations, methods=["POST"]
                ),
            ],
            exception_handlers={
                exc: errors.exception_handler for exc in errors.HANDLED_EXCEPTIONS
            },
        )
        # Above `exception_handlers` only triggers on exact exception classes.
        # This here is a fallback to add our custom headers in all other cases.
        app.add_exception_handler(Exception, errors.exception_handler)

        # Unregister default prometheus metrics collectors
        REGISTRY.unregister(process_collector.PROCESS_COLLECTOR)
        REGISTRY.unregister(platform_collector.PLATFORM_COLLECTOR)
        REGISTRY.unregister(gc_collector.GC_COLLECTOR)
        # Disable exporting _created metrics
        metrics.disable_created_metrics()
        # Add prometheus asgi middleware to route /metrics requests
        metrics_app = make_asgi_app()
        app.mount("/metrics", metrics_app)

        return app

    def start(self):
        log_level = (
            "DEBUG"
            if self._config["runtime"].get("enable_debug_logs", False)
            else "INFO"
        )
        cfg = uvicorn.Config(
            self.create_application(),
            # We hard-code the http parser as h11 (the default) in case the user has
            # httptools installed, which does not work with our requests & version
            # of uvicorn.
            http="h11",
            host="0.0.0.0",
            port=self._http_port,
            workers=1,
            timeout_graceful_shutdown=TIMEOUT_GRACEFUL_SHUTDOWN,
            log_config=log_config.make_log_config(log_level),
            ws_max_size=WS_MAX_MSG_SZ_BYTES,
        )
        cfg.setup_event_loop()  # Call this so uvloop gets used
        server = uvicorn.Server(config=cfg)
        self._server = server
        asyncio.run(server.serve())
