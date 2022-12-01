import json  # noqa: E402
import logging  # noqa: E402
from http import HTTPStatus  # noqa: E402
from threading import Thread
from typing import List

import numpy as np  # noqa: E402
import tornado.web  # noqa: E402
from kfserving import KFModel
from kfserving.handlers.http import HTTPHandler  # noqa: E402
from kfserving.kfserver import HealthHandler, KFServer, ListHandler
from pythonjsonlogger import jsonlogger  # noqa: E402
from truss.templates.server.common.lib_support import ensure_kfserving_installed
from truss.templates.server.common.serialization import DeepNumpyEncoder  # noqa: E402
from truss.templates.server.common.serialization import (  # noqa: E402
    truss_msgpack_deserialize,
    truss_msgpack_serialize,
)
from truss.templates.server.common.util import (  # noqa: E402
    assign_request_to_inputs_instances_after_validation,
)
from truss.templates.server.model_wrapper import ModelWrapper

ensure_kfserving_installed()

logger = logging.getLogger(__name__)


def _configure_logging():
    json_log_handler = logging.StreamHandler()
    json_log_handler.setFormatter(
        jsonlogger.JsonFormatter("%(asctime)s %(levelname)s %(message)s")
    )
    logger = logging.getLogger()
    for handler in logger.handlers:
        logger.removeHandler(handler)
    logger.addHandler(json_log_handler)


class TrussHTTPBinaryHandler(HTTPHandler):
    def validate(self, request):
        if (
            "instances" in request
            and not isinstance(request["instances"], (list, np.ndarray))
        ) or (
            "inputs" in request
            and not isinstance(request["inputs"], (list, np.ndarray))
        ):
            raise tornado.web.HTTPError(
                status_code=HTTPStatus.BAD_REQUEST,
                reason='Expected "instances" or "inputs" to be a list or NumPy ndarray',
            )
        return assign_request_to_inputs_instances_after_validation(request)


class TrussHTTPHandler(HTTPHandler):
    def validate(self, request):
        if ("instances" in request and not isinstance(request["instances"], list)) or (
            "inputs" in request and not isinstance(request["inputs"], list)
        ):
            raise tornado.web.HTTPError(
                status_code=HTTPStatus.BAD_REQUEST,
                reason='Expected "instances" or "inputs" to be a list',
            )
        return assign_request_to_inputs_instances_after_validation(request)


class TrussPredictHandler(TrussHTTPBinaryHandler):
    def post(self, name: str):
        model = self.get_model(name)
        try:
            body = truss_msgpack_deserialize(self.request.body)
        except Exception as e:
            raise tornado.web.HTTPError(
                status_code=HTTPStatus.BAD_REQUEST,
                reason="Unrecognized request format: %s" % e,
            )
        request = self.validate(body)
        request = model.preprocess(request)
        response = model.predict(request)
        response = model.postprocess(response)
        try:
            final_response = truss_msgpack_serialize(response)
        except Exception as e:
            raise tornado.web.HTTPError(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                reason="Unable to serialize model prediction to response: %s" % e,
            )
        self.write(final_response)


class TrussExplainHandler(TrussHTTPBinaryHandler):
    def post(self, name: str):
        model = self.get_model(name)
        try:
            body = truss_msgpack_deserialize(self.request.body)
        except Exception as e:
            raise tornado.web.HTTPError(
                status_code=HTTPStatus.BAD_REQUEST,
                reason="Unrecognized request format: %s" % e,
            )
        request = self.validate(body)
        request = model.preprocess(request)
        response = model.explain(request)
        response = model.postprocess(response)
        try:
            final_response = truss_msgpack_serialize(response)
        except Exception as e:
            raise tornado.web.HTTPError(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                reason="Unable to serialize model prediction to response: %s" % e,
            )
        self.write(final_response)


class PredictHandler(TrussHTTPHandler):
    def post(self, name: str):
        model = self.get_model(name)
        try:
            body = json.loads(self.request.body)
        except json.decoder.JSONDecodeError as e:
            raise tornado.web.HTTPError(
                status_code=HTTPStatus.BAD_REQUEST,
                reason="Unrecognized request format: %s" % e,
            )
        request = self.validate(body)
        request = model.preprocess(request)
        response = model.predict(request)
        response = model.postprocess(response)
        try:
            final_response = json.dumps(response, cls=DeepNumpyEncoder)
        except Exception as e:
            raise tornado.web.HTTPError(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                reason="Unable to serialize model prediction to response: %s" % e,
            )
        self.write(final_response)


class ExplainHandler(TrussHTTPHandler):
    def post(self, name: str):
        model = self.get_model(name)
        try:
            body = json.loads(self.request.body)
        except json.decoder.JSONDecodeError as e:
            raise tornado.web.HTTPError(
                status_code=HTTPStatus.BAD_REQUEST,
                reason="Unrecognized request format: %s" % e,
            )
        request = self.validate(body)
        request = model.preprocess(request)
        response = model.explain(request)
        response = model.postprocess(response)
        try:
            final_response = json.dumps(response, cls=DeepNumpyEncoder)
        except Exception as e:
            raise tornado.web.HTTPError(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                reason="Unable to serialize model prediction to response: %s" % e,
            )
        self.write(final_response)


class LivenessAndLoadHandler(TrussHTTPHandler):

    _load_thread: Thread = None

    @staticmethod
    def _exec_load(model):
        model.load()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get(self):
        for name in self.models:
            model = self.models[name]

            # if the model is ready or loading is in flight
            if not model.ready and not self._load_thread:
                self._load_thread = Thread(
                    target=LivenessAndLoadHandler._exec_load, args=[model]
                )
                self._load_thread.start()

            self.write("Alive")


class TrussServer(KFServer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _configure_logging()

    def start(self, models: List[KFModel]):
        if len(models) != 1:
            raise RuntimeError("TrussServer only supports one model")

        if not isinstance(models[0], ModelWrapper):
            raise ValueError(
                "TrussServer only accepts ModelWrapper instances instead of KFServer instances"
            )

        super().start(models)

    def create_application(self):
        return tornado.web.Application(
            [
                # Server Liveness API returns 200 if server is alive.
                (r"/", LivenessAndLoadHandler, dict(models=self.registered_models)),
                (r"/v1/models", ListHandler, dict(models=self.registered_models)),
                # Model Health API returns 200 if model is ready to serve.
                (
                    r"/v1/models/([a-zA-Z0-9_-]+)",
                    HealthHandler,
                    dict(models=self.registered_models),
                ),
                (
                    r"/v1/models/([a-zA-Z0-9_-]+):predict",
                    PredictHandler,
                    dict(models=self.registered_models),
                ),
                (
                    r"/v1/models/([a-zA-Z0-9_-]+):predict_binary",
                    TrussPredictHandler,
                    dict(models=self.registered_models),
                ),
                (
                    r"/v1/models/([a-zA-Z0-9_-]+):explain",
                    ExplainHandler,
                    dict(models=self.registered_models),
                ),
                (
                    r"/v1/models/([a-zA-Z0-9_-]+):explain_binary",
                    TrussExplainHandler,
                    dict(models=self.registered_models),
                ),
            ]
        )
