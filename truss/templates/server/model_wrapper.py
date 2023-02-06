import importlib
import inspect
import logging
import sys
import traceback
from enum import Enum
from pathlib import Path
from threading import Lock, Thread
from typing import Dict, Optional, Union

import kserve
import numpy as np

# TODO: add to requirements or ignore if not GPU
import torch
from cloudevents.http import CloudEvent
from common.util import assign_request_to_inputs_instances_after_validation
from kserve.errors import InvalidInput
from kserve.grpc.grpc_predict_v2_pb2 import ModelInferRequest, ModelInferResponse
from shared.secrets_resolver import SecretsResolver

MODEL_BASENAME = "model"


class ModelWrapper(kserve.Model):
    class Status(Enum):
        NOT_READY = 0
        LOADING = 1
        READY = 2
        FAILED = 3

    _config: dict
    _model: object
    _load_lock: Lock = Lock()
    _predict_lock: Lock = Lock()
    _status: Status = Status.NOT_READY
    _logger: logging.Logger
    _base_dir: Optional[Path]

    def __init__(self, config: dict, base_dir: Optional[Path] = None):
        super().__init__(MODEL_BASENAME)
        self._config = config
        self.logger = logging.getLogger(__name__)
        self._base_dir = base_dir

    def load(self) -> bool:
        if self.ready:
            return self.ready

        # if we are already loading, just pass; our container will return 503 while we're loading
        if not self._load_lock.acquire(blocking=False):
            return False

        self._status = ModelWrapper.Status.LOADING

        self.logger.info("Executing model.load()...")

        try:
            self.try_load()
            self.ready = True
            self._status = ModelWrapper.Status.READY

            self.logger.info("Completed model.load() execution")

            return self.ready
        except Exception:
            self.logger.exception("Exception while loading model")
            self._status = ModelWrapper.Status.FAILED
        finally:
            self._load_lock.release()

        return self.ready

    def start_load(self):
        if self.should_load():
            thread = Thread(target=self.load)
            thread.start()

    def load_failed(self) -> bool:
        return self._status == ModelWrapper.Status.FAILED

    def should_load(self) -> bool:
        # don't retry failed loads
        return (
            not self._load_lock.locked()
            and not self._status == ModelWrapper.Status.FAILED
            and not self.ready
        )

    def try_load(self):
        # TODO: Handle bundled packages properly for multi config
        if "bundled_packages_dir" in self._config:
            bundled_packages_path = Path("/packages")
            if bundled_packages_path.exists():
                sys.path.append(str(bundled_packages_path))
        model_module_name = str(
            Path(self._config["model_class_filename"]).with_suffix("")
        )
        if self._base_dir:
            spec = importlib.util.spec_from_file_location(
                model_module_name,
                self._base_dir
                / self._config["model_module_dir"]
                / self._config["model_class_filename"],
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            model_class = getattr(module, self._config["model_class_name"])
        else:
            module = importlib.import_module(
                f"{self._config['model_module_dir']}.{model_module_name}"
            )
            model_class = getattr(module, self._config["model_class_name"])
        model_class_signature = inspect.signature(model_class)
        model_init_params = {}
        if _signature_accepts_keyword_arg(model_class_signature, "config"):
            model_init_params["config"] = self._config
        if _signature_accepts_keyword_arg(model_class_signature, "data_dir"):
            model_init_params["data_dir"] = Path("data")
        if _signature_accepts_keyword_arg(model_class_signature, "secrets"):
            model_init_params["secrets"] = SecretsResolver.get_secrets(self._config)
        self._model = model_class(**model_init_params)

        if hasattr(self._model, "load"):
            self._model.load()

    def validate(self, payload):
        if (
            "instances" in payload
            and not isinstance(payload["instances"], (list, np.ndarray))
            or "inputs" in payload
            and not isinstance(payload["inputs"], (list, np.ndarray))
        ):
            raise InvalidInput(
                'Expected "instances" or "inputs" to be a list or NumPy ndarray'
            )

        return assign_request_to_inputs_instances_after_validation(payload)

    def preprocess(
        self,
        payload: Union[Dict, CloudEvent, ModelInferRequest],
        headers: Dict[str, str] = None,
    ) -> Union[Dict, ModelInferRequest]:
        if not hasattr(self._model, "preprocess"):
            return payload
        return self._model.preprocess(payload)

    def prepare(self):
        if not hasattr(self._model, "prepare"):
            return
        # Force re-calc GPU memory
        torch.cuda.memory_allocated()
        self._model.prepare()

    def standby(self):
        if not hasattr(self._model, "standby"):
            return
        self._model.standby()
        # Force re-calc GPU memory
        torch.cuda.memory_allocated()

    def postprocess(
        self, response: Union[Dict, ModelInferResponse], headers: Dict[str, str] = None
    ) -> Dict:
        if not hasattr(self._model, "postprocess"):
            return response
        return self._model.postprocess(response)

    def predict(
        self, payload: Union[Dict, ModelInferRequest], headers: Dict[str, str] = None
    ) -> Union[Dict, ModelInferResponse]:
        try:
            self._predict_lock.acquire()
            self.prepare()
            return self._model.predict(payload)
        except Exception:
            response = {}
            logging.exception("Exception while running predict")
            response["error"] = {"traceback": traceback.format_exc()}
            return response
        finally:
            self.standby()
            self._predict_lock.release()


def _signature_accepts_keyword_arg(signature: inspect.Signature, kwarg: str) -> bool:
    return kwarg in signature.parameters or _signature_accepts_kwargs(signature)


def _signature_accepts_kwargs(signature: inspect.Signature) -> bool:
    for param in signature.parameters.values():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            return True
    return False
