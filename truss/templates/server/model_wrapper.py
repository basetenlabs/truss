import importlib
import inspect
import logging
import os
import sys
import time
import traceback
from collections.abc import Generator
from enum import Enum
from pathlib import Path
from threading import Lock, Thread
from typing import Any, Dict, Optional, Union

from common.patches import apply_patches
from common.retry import retry
from shared.secrets_resolver import SecretsResolver

MODEL_BASENAME = "model"

NUM_LOAD_RETRIES = int(os.environ.get("NUM_LOAD_RETRIES_TRUSS", "3"))


class ModelWrapper:
    class Status(Enum):
        NOT_READY = 0
        LOADING = 1
        READY = 2
        FAILED = 3

    def __init__(self, config: Dict):
        self._config = config
        self._logger = logging.getLogger()
        self.name = MODEL_BASENAME
        self.ready = False
        self._load_lock = Lock()
        self._predict_lock = Lock()
        self._status = ModelWrapper.Status.NOT_READY

    def load(self) -> bool:
        if self.ready:
            return self.ready

        # if we are already loading, just pass; our container will return 503 while we're loading
        if not self._load_lock.acquire(blocking=False):
            return False

        self._status = ModelWrapper.Status.LOADING

        self._logger.info("Executing model.load()...")

        try:
            start_time = time.perf_counter()
            self.try_load()
            self.ready = True
            self._status = ModelWrapper.Status.READY
            self._logger.info(
                f"Completed model.load() execution in {_elapsed_ms(start_time)} ms"
            )

            return self.ready
        except Exception:
            self._logger.exception("Exception while loading model")
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
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)

        if "bundled_packages_dir" in self._config:
            bundled_packages_path = Path("/packages")
            if bundled_packages_path.exists():
                sys.path.append(str(bundled_packages_path))
        model_module_name = str(
            Path(self._config["model_class_filename"]).with_suffix("")
        )
        module = importlib.import_module(
            f"{self._config['model_module_dir']}.{model_module_name}"
        )
        model_class = getattr(module, self._config["model_class_name"])
        model_class_signature = inspect.signature(model_class)
        model_init_params = {}
        if _signature_accepts_keyword_arg(model_class_signature, "config"):
            model_init_params["config"] = self._config
        if _signature_accepts_keyword_arg(model_class_signature, "data_dir"):
            model_init_params["data_dir"] = data_dir
        if _signature_accepts_keyword_arg(model_class_signature, "secrets"):
            model_init_params["secrets"] = SecretsResolver.get_secrets(self._config)
        apply_patches(
            self._config.get("apply_library_patches", True),
            self._config["requirements"],
        )
        self._model = model_class(**model_init_params)

        if hasattr(self._model, "load"):
            retry(
                self._model.load,
                NUM_LOAD_RETRIES,
                self._logger.warn,
                "Failed to load model.",
                gap_seconds=1.0,
            )

    def preprocess(
        self,
        payload: Any,
        headers: Optional[Dict[str, str]] = None,
    ) -> Any:
        if not hasattr(self._model, "preprocess"):
            return payload
        return self._model.preprocess(payload)  # type: ignore

    def postprocess(
        self,
        response: Any,
        headers: Optional[Dict[str, str]] = None,
    ) -> Any:
        if not hasattr(self._model, "postprocess"):
            return response
        return self._model.postprocess(response)  # type: ignore

    def predict(
        self,
        payload: Any,
        headers: Optional[Dict[str, str]] = None,
    ) -> Any:
        try:
            self._predict_lock.acquire()
            return self._model.predict(payload)  # type: ignore
        except Exception:
            response = {}
            logging.exception("Exception while running predict")
            response["error"] = {"traceback": traceback.format_exc()}
            return response
        finally:
            self._predict_lock.release()

    async def __call__(
        self, body: Any, headers: Optional[Dict[str, str]] = None
    ) -> Union[Dict, Generator]:
        """Method to call predictor or explainer with the given input.

        Args:
            body (Any): Request payload body.
            headers (Dict): Request headers.

        Returns:
            Dict: Response output from preprocess -> predictor -> postprocess
        """

        payload = (
            await self.preprocess(body, headers)
            if inspect.iscoroutinefunction(self.preprocess)
            else self.preprocess(body, headers)
        )

        response = (
            (await self.predict(payload, headers))
            if inspect.iscoroutinefunction(self.predict)
            else self.predict(payload, headers)
        )

        response = self.postprocess(response, headers)

        if isinstance(response, Generator):
            # In the case of streaming responses, we need to:
            #   1. Check the 'Accept' header. If the accept type is
            #      "application/json" , consume the whole stream and return it,
            #      Else, return the stream.
            #   2. For streaming responses, the predict lock needs to properly
            #      wrap the whole generated response. To achieve that, we construct
            #      a new generator that does fully
            #      wrap the entire streaming response.
            locked_response_generator = _locked_response_generator(
                response, self._predict_lock
            )

            if headers and headers.get("accept") == "application/json":
                response = _convert_streamed_response_to_string(
                    locked_response_generator
                )
            else:
                response = locked_response_generator

        return response


def _locked_response_generator(response: Any, lock: Lock):
    with lock:
        for chunk in response:
            yield chunk


def _convert_streamed_response_to_string(response: Any):
    return "".join([str(chunk) for chunk in list(response)])


def _signature_accepts_keyword_arg(signature: inspect.Signature, kwarg: str) -> bool:
    return kwarg in signature.parameters or _signature_accepts_kwargs(signature)


def _signature_accepts_kwargs(signature: inspect.Signature) -> bool:
    for param in signature.parameters.values():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            return True
    return False


def _elapsed_ms(since_micro_seconds: float) -> int:
    return int((time.perf_counter() - since_micro_seconds) * 1000)
