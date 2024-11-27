import asyncio
import json
import logging
import os
import signal
import sys
from http import HTTPStatus
from pathlib import Path
from typing import Dict, Optional, Union

import pydantic
import uvicorn
import yaml
from common import errors, tracing
from common.schema import TrussSchema
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import ORJSONResponse, StreamingResponse
from fastapi.routing import APIRoute as FastAPIRoute
from model_wrapper import ModelWrapper
from opentelemetry import propagate as otel_propagate
from opentelemetry import trace
from opentelemetry.sdk import trace as sdk_trace
from shared import serialization
from shared.logging import setup_logging
from shared.secrets_resolver import SecretsResolver
from starlette.requests import ClientDisconnect
from starlette.responses import Response

if sys.version_info >= (3, 9):
    from typing import AsyncGenerator, Generator
else:
    from typing_extensions import AsyncGenerator, Generator

# [IMPORTANT] A lot of things depend on this currently, change with extreme care.
TIMEOUT_GRACEFUL_SHUTDOWN = 120
INFERENCE_SERVER_FAILED_FILE = Path("~/inference_server_crashed.txt").expanduser()
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


async def parse_body(request: Request) -> bytes:
    """
    Used by FastAPI to read body in an asynchronous manner
    """
    try:
        return await request.body()
    except ClientDisconnect as exc:
        error_message = "Client disconnected"
        logging.error(error_message)
        raise HTTPException(status_code=499, detail=error_message) from exc


class BasetenEndpoints:
    """The implementation of the model server endpoints.

    Historically, we relied on the kserve server interface, which assumes that
    multiple models are running behind a registry. As a result, some arguments to
    to functions will rename unused except for backwards compatibility checks.
    """

    def __init__(self, model: ModelWrapper, tracer: sdk_trace.Tracer) -> None:
        self._model = model
        self._tracer = tracer

    def _safe_lookup_model(self, model_name: str) -> ModelWrapper:
        if model_name != self._model.name:
            raise errors.ModelMissingError(model_name)
        return self._model

    @staticmethod
    def check_healthy(model: ModelWrapper):
        if model.load_failed:
            INFERENCE_SERVER_FAILED_FILE.touch()
            os.kill(os.getpid(), signal.SIGKILL)

        if not model.ready:
            raise errors.ModelNotReady(model.name)

    async def model_ready(self, model_name: str) -> Dict[str, Union[str, bool]]:
        self.check_healthy(self._safe_lookup_model(model_name))

        return {}

    async def invocations_ready(self) -> Dict[str, Union[str, bool]]:
        """
        This method provides compatibility with Sagemaker hosting for the 'ping' endpoint.
        """
        if self._model is None:
            raise errors.ModelMissingError("model")
        self.check_healthy(self._model)

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
    ) -> serialization.InputType:
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

    async def predict(
        self, model_name: str, request: Request, body_raw: bytes = Depends(parse_body)
    ) -> Response:
        """
        This method calls the user-provided predict method
        """
        if await request.is_disconnected():
            msg = "Client disconnected. Skipping `predict`."
            logging.info(msg)
            raise ClientDisconnect(msg)

        model: ModelWrapper = self._safe_lookup_model(model_name)

        self.check_healthy(model)
        trace_ctx = otel_propagate.extract(request.headers) or None
        # This is the top-level span in the truss-server, so we set the context here.
        # Nested spans "inherit" context automatically.
        with self._tracer.start_as_current_span(
            "predict-endpoint", context=trace_ctx
        ) as span:
            inputs: Optional[serialization.InputType]
            if model.model_descriptor.skip_input_parsing:
                inputs = None
            else:
                inputs = await self._parse_body(
                    request, body_raw, model.model_descriptor.truss_schema, span
                )
            # Calls ModelWrapper which runs: preprocess, predict, postprocess.
            with tracing.section_as_event(span, "model-call"):
                result: Union[Dict, Generator] = await model(inputs, request)

            # In the case that the model returns a Generator object, return a
            # StreamingResponse instead.
            if isinstance(result, (AsyncGenerator, Generator)):
                # media_type in StreamingResponse sets the Content-Type header
                return StreamingResponse(result, media_type="application/octet-stream")
            elif isinstance(result, Response):
                if result.status_code >= HTTPStatus.MULTIPLE_CHOICES.value:
                    errors.add_error_headers_to_user_response(result)
                return result

            response_headers = {}
            if self.is_binary(request):
                with tracing.section_as_event(span, "binary-serialize"):
                    response_headers["Content-Type"] = "application/octet-stream"
                    return Response(
                        content=serialization.truss_msgpack_serialize(result),
                        headers=response_headers,
                    )
            else:
                with tracing.section_as_event(span, "json-serialize"):
                    response_headers["Content-Type"] = "application/json"
                    return Response(
                        content=json.dumps(result, cls=serialization.DeepNumpyEncoder),
                        headers=response_headers,
                    )

    async def schema(self, model_name: str) -> Dict:
        model: ModelWrapper = self._safe_lookup_model(model_name)
        if model.model_descriptor.truss_schema is None:
            # If there is not a TrussSchema, we return a 404.
            if model.ready:
                raise HTTPException(status_code=404, detail="No schema found")
            else:
                raise HTTPException(
                    status_code=503,
                    detail="Schema not available, please try again later.",
                )
        else:
            return model.model_descriptor.truss_schema.serialize()

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

    def __init__(
        self,
        http_port: int,
        config_or_path: Union[str, Path, Dict],
        setup_json_logger: bool = True,
    ):
        if isinstance(config_or_path, (str, Path)):
            with open(config_or_path, encoding="utf-8") as config_file:
                config = yaml.safe_load(config_file)
        else:
            config = config_or_path

        secrets = SecretsResolver.get_secrets(config)
        tracer = tracing.get_truss_tracer(secrets, config)
        self._setup_json_logger = setup_json_logger
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
        if self._setup_json_logger:
            setup_logging()
        self._model.start_load_thread()
        asyncio.create_task(self._shutdown_if_load_fails())
        self._model.setup_polling_for_environment_updates()

    async def _shutdown_if_load_fails(self):
        while not self._model.ready:
            await asyncio.sleep(0.5)
            if self._model.load_failed:
                assert self._server is not None
                logging.info("Trying shut down.")
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
                # Endpoint aliases for Sagemaker hosting
                FastAPIRoute(r"/ping", self._endpoints.invocations_ready),
                FastAPIRoute(
                    r"/invocations",
                    self._endpoints.invocations,
                    methods=["POST"],
                ),
            ],
            exception_handlers={
                exc: errors.exception_handler for exc in errors.HANDLED_EXCEPTIONS
            },
        )
        # Above `exception_handlers` only triggers on exact exception classes.
        # This here is a fallback to add our custom headers in all other cases.
        app.add_exception_handler(Exception, errors.exception_handler)

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
            log_config={
                "version": 1,
                "formatters": {
                    "default": {
                        "()": "uvicorn.logging.DefaultFormatter",
                        "datefmt": DATE_FORMAT,
                        "fmt": "%(asctime)s.%(msecs)03d %(name)s %(levelprefix)s %(message)s",
                        "use_colors": None,
                    },
                    "access": {
                        "()": "uvicorn.logging.AccessFormatter",
                        "datefmt": DATE_FORMAT,
                        "fmt": "%(asctime)s.%(msecs)03d %(name)s %(levelprefix)s %(client_addr)s %(process)s - "
                        '"%(request_line)s" %(status_code)s',
                        # noqa: E501
                    },
                },
                "handlers": {
                    "default": {
                        "formatter": "default",
                        "class": "logging.StreamHandler",
                        "stream": "ext://sys.stderr",
                    },
                    "access": {
                        "formatter": "access",
                        "class": "logging.StreamHandler",
                        "stream": "ext://sys.stdout",
                    },
                },
                "loggers": {
                    "uvicorn": {"handlers": ["default"], "level": log_level},
                    "uvicorn.error": {"level": "INFO"},
                    "uvicorn.access": {
                        "handlers": ["access"],
                        "level": "INFO",
                        "propagate": False,
                    },
                },
            },
        )
        cfg.setup_event_loop()  # Call this so uvloop gets used
        server = uvicorn.Server(config=cfg)
        self._server = server
        asyncio.run(server.serve())
