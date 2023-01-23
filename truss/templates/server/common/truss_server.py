import json
import logging
from typing import Dict, Optional, Union

import kserve
import kserve.errors as errors
import numpy as np
from common.serialization import (
    DeepNumpyEncoder,
    truss_msgpack_deserialize,
    truss_msgpack_serialize,
)
from common.util import assign_request_to_inputs_instances_after_validation
from fastapi import Depends, FastAPI, Request
from fastapi.responses import ORJSONResponse
from fastapi.routing import APIRoute as FastAPIRoute
from kserve.errors import InvalidInput
from kserve.handlers import DataPlane, ModelRepositoryExtension, V1Endpoints
from model_wrapper import ModelWrapper
from starlette.responses import Response

logger = logging.getLogger(__name__)


async def parse_body(request: Request):
    data: bytes = await request.body()
    return data


class BasetenEndpoints(V1Endpoints):
    def __init__(
        self,
        dataplane: DataPlane,
        model_repository_extension: Optional[ModelRepositoryExtension] = None,
    ):
        super().__init__(dataplane, model_repository_extension)

    @staticmethod
    def check_healthy(model: ModelWrapper):
        model.start_load()

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
        model: ModelWrapper = self.dataplane.get_model_from_registry(model_name)

        self.check_healthy(model)

        body: dict
        if self.is_binary(request):
            body = truss_msgpack_deserialize(body_raw)
        else:
            body = json.loads(body_raw)

        body = assign_request_to_inputs_instances_after_validation(body)

        if (
            "instances" in request
            and not isinstance(request["instances"], (list, np.ndarray))
            or "inputs" in request
            and not isinstance(request["inputs"], (list, np.ndarray))
        ):
            raise InvalidInput(
                'Expected "instances" or "inputs" to be a list or NumPy ndarray'
            )

        response = model.predict(body, headers=dict(request.headers.items()))

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


class TrussServer(kserve.ModelServer):

    _endpoints: BasetenEndpoints
    _model: ModelWrapper

    def __init__(self, http_port: int, model: ModelWrapper):
        super().__init__(
            http_port=http_port,
            enable_grpc=False,
            workers=1,
            enable_docs_url=False,
            enable_latency_logging=False,
        )

        self._model = model
        self._endpoints = BasetenEndpoints(
            self.dataplane, self.model_repository_extension
        )

    def start_model(self) -> None:
        """
        Overloaded version of super().start to use instance model in TrussServer
        """
        super().start([self._model])

    def create_application(self):
        return FastAPI(
            title="Baseten Inference Server",
            docs_url=None,
            redoc_url=None,
            default_response_class=ORJSONResponse,
            on_startup=[self._model.start_load],
            routes=[
                # liveness endpoint
                FastAPIRoute(r"/", self.dataplane.live),
                # readiness endpoint
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
