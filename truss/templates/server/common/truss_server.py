import asyncio
import json
import logging
import multiprocessing
import os
import signal
import socket
import sys
import time
from collections.abc import Generator
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Optional, Union

import common.errors as errors
import shared.util as utils
import uvicorn
from common.termination_handler_middleware import TerminationHandlerMiddleware
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import ORJSONResponse, StreamingResponse
from fastapi.routing import APIRoute as FastAPIRoute
from model_wrapper import ModelWrapper
from shared.logging import setup_logging
from shared.serialization import (
    DeepNumpyEncoder,
    truss_msgpack_deserialize,
    truss_msgpack_serialize,
)
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import ClientDisconnect
from starlette.responses import Response

# [IMPORTANT] A lot of things depend on this currently.
# Please consider the following when increasing this:
# 1. Self-termination on model load fail.
# 2. Graceful termination.
DEFAULT_NUM_WORKERS = 1
DEFAULT_NUM_SERVER_PROCESSES = 1
WORKER_TERMINATION_TIMEOUT_SECS = 120.0
WORKER_TERMINATION_CHECK_INTERVAL_SECS = 0.5


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


FORMAT = "%(asctime)s.%(msecs)03d %(name)s %(levelname)s [%(funcName)s():%(lineno)s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
INFERENCE_SERVER_FAILED_FILE = Path("~/inference_server_crashed.txt").expanduser()
logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt=DATE_FORMAT)


class UvicornCustomServer(multiprocessing.Process):
    def __init__(
        self, config: uvicorn.Config, sockets: Optional[List[socket.socket]] = None
    ):
        super().__init__()
        self.sockets = sockets
        self.config = config

    def stop(self):
        self.terminate()

    def run(self):
        server = uvicorn.Server(config=self.config)
        asyncio.run(server.serve(sockets=self.sockets))


class BasetenEndpoints:
    """The implementation of the model server endpoints.

    Historically, we relied on the kserve server interface, which assumes that
    multiple models are running behind a registry. As a result, some arguments to
    to functions will rename unused except for backwards compatibility checks.
    """

    def __init__(self, model: ModelWrapper) -> None:
        self._model = model

    def _safe_lookup_model(self, model_name: str) -> ModelWrapper:
        if model_name != self._model.name:
            raise errors.ModelMissingError(model_name)
        return self._model

    @staticmethod
    def check_healthy(model: ModelWrapper):
        if model.load_failed():
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

    async def predict(
        self, model_name: str, request: Request, body_raw: bytes = Depends(parse_body)
    ) -> Response:
        """
        This method calls the user-provided predict method
        """
        model: ModelWrapper = self._safe_lookup_model(model_name)

        self.check_healthy(model)

        body: Dict
        if self.is_binary(request):
            body = truss_msgpack_deserialize(body_raw)
        else:
            try:
                body = json.loads(body_raw)
            except json.JSONDecodeError as e:
                error_message = f"Invalid JSON payload: {str(e)}"
                logging.error(error_message)
                raise HTTPException(status_code=400, detail=error_message)

        # calls ModelWrapper.__call__, which runs validate, preprocess, predict, and postprocess
        response: Union[Dict, Generator] = await model(
            body,
            headers=utils.transform_keys(request.headers, lambda key: key.lower()),
        )

        # In the case that the model returns a Generator object, return a
        # StreamingResponse instead.
        if isinstance(response, (AsyncGenerator, Generator)):
            # media_type in StreamingResponse sets the Content-Type header
            return StreamingResponse(response, media_type="application/octet-stream")

        response_headers = {}
        if self.is_binary(request):
            response_headers["Content-Type"] = "application/octet-stream"
            return Response(
                content=truss_msgpack_serialize(response), headers=response_headers
            )
        else:
            response_headers["Content-Type"] = "application/json"
            return Response(
                content=json.dumps(response, cls=DeepNumpyEncoder),
                headers=response_headers,
            )

    async def schema(self, model_name: str) -> Dict:
        model: ModelWrapper = self._safe_lookup_model(model_name)

        if model.truss_schema is None:
            # If there is not a TrussSchema, we return a 404.

            if model.ready:
                raise HTTPException(status_code=404, detail="No schema found")
            else:
                raise HTTPException(
                    status_code=503,
                    detail="Schema not available, please try again later.",
                )
        else:
            return model.truss_schema.serialize()

    @staticmethod
    def is_binary(request: Request):
        return (
            "Content-Type" in request.headers
            and request.headers["Content-Type"] == "application/octet-stream"
        )


class TrussServer:
    """This wrapper class manages creation and cleanup of uvicorn server processes running the FastAPI inference server app

    TrussServer runs as a main process managing UvicornCustomServer subprocesses that in turn may manage
    their own worker processes. Notably, this main process is kept alive when running `servers_task()`
    because of the child uvicorn server processes' main loop.

    """

    def __init__(
        self,
        http_port: int,
        config: Dict,
        setup_json_logger: bool = True,
    ):
        self.http_port = http_port
        self._config = config
        self._model = ModelWrapper(self._config)
        self._endpoints = BasetenEndpoints(self._model)
        self._setup_json_logger = setup_json_logger

    def cleanup(self):
        if INFERENCE_SERVER_FAILED_FILE.exists():
            INFERENCE_SERVER_FAILED_FILE.unlink()

    def on_startup(self):
        """
        This method will be started inside the main process, so here is where we want to setup our logging and model
        """
        self.cleanup()

        if self._setup_json_logger:
            setup_logging()

        self._model.start_load()

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
                errors.InferenceError: errors.inference_error_handler,
                errors.ModelNotFound: errors.model_not_found_handler,
                errors.ModelNotReady: errors.model_not_ready_handler,
                NotImplementedError: errors.not_implemented_error_handler,
                HTTPException: errors.http_exception_handler,
                Exception: errors.generic_exception_handler,
            },
        )

        def exit_self():
            # Note that this kills the current process, the worker process, not
            # the main truss_server process.
            utils.kill_child_processes(os.getpid())
            sys.exit()

        termination_handler_middleware = TerminationHandlerMiddleware(
            on_stop=lambda: None,
            on_term=exit_self,
        )
        app.add_middleware(BaseHTTPMiddleware, dispatch=termination_handler_middleware)
        return app

    def start(self):
        cfg = uvicorn.Config(
            self.create_application(),
            # We hard-code the http parser as h11 (the default) in case the user has
            # httptools installed, which does not work with our requests & version
            # of uvicorn.
            http="h11",
            host="0.0.0.0",
            port=self.http_port,
            workers=DEFAULT_NUM_WORKERS,
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
                    "uvicorn": {"handlers": ["default"], "level": "INFO"},
                    "uvicorn.error": {"level": "INFO"},
                    "uvicorn.access": {
                        "handlers": ["access"],
                        "level": "INFO",
                        "propagate": False,
                    },
                },
            },
        )

        # Call this so uvloop gets used
        cfg.setup_event_loop()

        async def serve():
            serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            serversocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            serversocket.bind((cfg.host, cfg.port))
            serversocket.listen(5)

            num_server_procs = self._config.get("runtime", {}).get(
                "num_workers", DEFAULT_NUM_SERVER_PROCESSES
            )
            logging.info(f"starting {num_server_procs} uvicorn server processes")
            servers: List[UvicornCustomServer] = []
            for _ in range(num_server_procs):
                server = UvicornCustomServer(config=cfg, sockets=[serversocket])
                server.start()
                servers.append(server)

            def stop_servers():
                # Send stop signal, then wait for all to exit
                for server in servers:
                    # Sends term signal to the process, which should be handled
                    # by the termination handler.
                    server.stop()

                termination_check_attempts = int(
                    WORKER_TERMINATION_TIMEOUT_SECS
                    / WORKER_TERMINATION_CHECK_INTERVAL_SECS
                )
                for _ in range(termination_check_attempts):
                    time.sleep(WORKER_TERMINATION_CHECK_INTERVAL_SECS)
                    if utils.all_processes_dead(servers):
                        return

            for sig in [signal.SIGINT, signal.SIGTERM, signal.SIGQUIT]:
                signal.signal(sig, lambda sig, frame: stop_servers())

        async def servers_task():
            servers = [serve()]
            await asyncio.gather(*servers)

        asyncio.run(servers_task())
