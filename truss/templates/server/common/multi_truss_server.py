import asyncio
import json
from pathlib import Path
from typing import Dict, Optional, Union

import kserve
import kserve.errors as errors
import yaml
from common.logging import setup_logging
from common.serialization import (
    DeepNumpyEncoder,
    truss_msgpack_deserialize,
    truss_msgpack_serialize,
)
from fastapi import Depends, FastAPI, Request
from fastapi.responses import ORJSONResponse
from fastapi.routing import APIRoute as FastAPIRoute
from kserve.handlers import DataPlane, ModelRepositoryExtension, V1Endpoints
from kserve.model_repository import ModelRepository
from model_wrapper import ModelWrapper
from starlette.responses import Response


async def parse_body(request: Request) -> bytes:
    """
    Used by FastAPI to read body in an asynchronous manner
    """
    return await request.body()


class BasetenEndpoints(V1Endpoints):
    def __init__(
        self,
        dataplane: DataPlane,
        model_repository_extension: Optional[ModelRepositoryExtension] = None,
    ):
        super().__init__(dataplane, model_repository_extension)

    @staticmethod
    def check_healthy(model: ModelWrapper):
        if model.load_failed():
            raise errors.InferenceError("Model load failed")

        if not model.ready:
            raise errors.ModelNotReady(model.name)

    async def model_ready(self, model_name: str) -> Dict[str, Union[str, bool]]:
        model: ModelWrapper = self.dataplane.get_model_from_registry(model_name)

        self.check_healthy(model)

        return {}

    def predict(
        self, model_name: str, request: Request, body_raw: bytes = Depends(parse_body)
    ) -> Response:
        """
        This method is called by FastAPI, which introspects that it's not async, and schedules it on a thread
        """
        model: ModelWrapper = self.dataplane.get_model_from_registry(model_name)

        self.check_healthy(model)

        body: dict
        if self.is_binary(request):
            body = truss_msgpack_deserialize(body_raw)
        else:
            body = json.loads(body_raw)

        # calls kserve.model.Model.__call__, which runs validate, preprocess, predict, and postprocess
        response: dict = asyncio.run(model(body, headers=dict(request.headers.items())))

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

    @staticmethod
    def is_binary(request: Request):
        return (
            "Content-Type" in request.headers
            and request.headers["Content-Type"] == "application/octet-stream"
        )


class MultiTrussModelRepository(ModelRepository):
    def __init__(self, truss_configs_by_name: dict[str, dict], base_dir: Path):
        self.models: Dict[str, ModelWrapper] = {}
        self._configs_by_name = truss_configs_by_name
        self._base_dir = base_dir

    def load_models(self):
        for name, config in self._configs_by_name.items():
            print(f"Loading: {name}")
            self.models[name] = ModelWrapper(config, self._base_dir / name)
            self.models[name].start_load()

    def load(self, name: str) -> bool:
        self.models[name].start_load()
        return self.models[name].ready

    def load_model(self, name: str) -> bool:
        self.models[name].start_load()
        return self.models[name].ready


class MultiTrussServer(kserve.ModelServer):

    _endpoints: BasetenEndpoints
    _model_repository: MultiTrussModelRepository
    _config: dict
    _model_truss_dir: Path

    def _make_config_by_name(self, model_truss_dir):
        result = {}
        for truss_ref in self._config["trusses"]:
            with open(model_truss_dir / truss_ref["name"] / "config.yaml", "r") as f:
                result[truss_ref["name"]] = yaml.safe_load(f)
        return result

    def __init__(self, http_port: int, multi_truss_config: dict, model_truss_dir: Path):
        self._config = multi_truss_config
        self._model_repository = MultiTrussModelRepository(
            truss_configs_by_name=self._make_config_by_name(model_truss_dir),
            base_dir=model_truss_dir,
        )
        super().__init__(
            http_port=http_port,
            enable_grpc=False,
            workers=1,
            enable_docs_url=False,
            enable_latency_logging=False,
            registered_models=self._model_repository,
        )

        self._endpoints = BasetenEndpoints(
            self.dataplane, self.model_repository_extension
        )

    def start_model(self) -> None:
        """
        Overloaded version of super().start to use instance model in TrussServer
        """
        super().start([])

    def on_startup(self):
        """
        This method will be started inside the main process, so here is where we want to setup our logging and model
        """
        setup_logging()
        self._model_repository.load_models()

    def create_application(self):
        return FastAPI(
            title="MultiTruss Inference Server",
            docs_url=None,
            redoc_url=None,
            default_response_class=ORJSONResponse,
            on_startup=[self.on_startup],
            routes=[
                # liveness endpoint
                FastAPIRoute(r"/", self.dataplane.live),
                # readiness endpoint. Handles model by name.
                FastAPIRoute(
                    r"/v1/models/{model_name}", self._endpoints.model_ready, tags=["V1"]
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
                FastAPIRoute(
                    r"/v1/models/{model_name}:explain",
                    self._endpoints.explain,
                    methods=["POST"],
                    tags=["V1"],
                ),
            ],
            exception_handlers={
                errors.InvalidInput: errors.invalid_input_handler,
                errors.InferenceError: errors.inference_error_handler,
                errors.ModelNotFound: errors.model_not_found_handler,
                errors.ModelNotReady: errors.model_not_ready_handler,
                NotImplementedError: errors.not_implemented_error_handler,
                Exception: errors.generic_exception_handler,
            },
        )
