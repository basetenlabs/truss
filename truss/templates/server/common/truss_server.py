import json  # noqa: E402
import logging  # noqa: E402
import os
import sys
from http import HTTPStatus  # noqa: E402
from threading import Thread
from typing import List

import numpy as np  # noqa: E402
import tornado.web  # noqa: E402
from common.lib_support import ensure_kfserving_installed
from common.serialization import DeepNumpyEncoder  # noqa: E402
from common.serialization import (  # noqa: E402
    truss_msgpack_deserialize,
    truss_msgpack_serialize,
)
from common.util import (  # noqa: E402
    assign_request_to_inputs_instances_after_validation,
)
from kfserving import KFModel
from kfserving.handlers.http import HTTPHandler  # noqa: E402
from kfserving.kfserver import HealthHandler, KFServer, ListHandler, LivenessHandler
from model_wrapper import ModelWrapper
from pythonjsonlogger import jsonlogger  # noqa: E402
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop

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


class TrussServer(KFServer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _configure_logging()

    def load_all(self, main_loop: IOLoop):
        try:
            for model in self.registered_models.get_models():
                model.load()
        except Exception as e:
            logging.error(f"Error loading model: {e}")

            # fixme(zack) for live reload, we don't want to kill the process?
            if not os.environ.get("CONTROL_SERVER_PORT"):
                self._http_server.stop()
                main_loop.stop()
                sys.exit(1)

    def start(self, models: List[KFModel], nest_asyncio: bool = False):
        if len(models) != 1:
            raise RuntimeError("TrussServer only supports one model")

        if not isinstance(models[0], ModelWrapper):
            raise ValueError(
                "TrussServer only accepts ModelWrapper instances instead of KFServer instances"
            )

        for model in models:
            self.register_model(model)

        self._http_server = HTTPServer(
            self.create_application(), max_buffer_size=self.max_buffer_size
        )

        logging.info("Listening on port %s", self.http_port)
        self._http_server.bind(self.http_port)

        logging.info("Will fork %d workers", self.workers)
        self._http_server.start(self.workers)

        Thread(
            target=self.load_all,
            args=[IOLoop.current()],
        ).start()

        IOLoop.current().start()

    def create_application(self):
        return tornado.web.Application(
            [
                # Server Liveness API returns 200 if server is alive.
                (r"/", LivenessHandler),
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
