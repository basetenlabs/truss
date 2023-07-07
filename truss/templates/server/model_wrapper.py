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
from queue import Queue
from threading import Lock, Thread
from typing import Any, Dict, Optional, Union

from anyio import to_thread
from common.patches import apply_patches
from common.retry import retry
from shared.secrets_resolver import SecretsResolver

MODEL_BASENAME = "model"

NUM_LOAD_RETRIES = int(os.environ.get("NUM_LOAD_RETRIES_TRUSS", "3"))
STREAMING_RESPONSE_QUEUE_READ_TIMEOUT_SECS = 60


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
            return self._model.predict(payload)  # type: ignore
        except Exception:
            logging.exception("Exception while running predict")
            return {"error": {"traceback": traceback.format_exc()}}

    async def __call__(
        self, body: Any, headers: Optional[Dict[str, str]] = None
    ) -> Union[Dict, Generator]:
        """Method to call predictor or explainer with the given input.

        Args:
            body (Any): Request payload body.
            headers (Dict): Request headers.

        Returns:
            Dict: Response output from preprocess -> predictor -> postprocess
            Generator: In case of streaming response
        """

        payload = (
            await self.preprocess(body, headers)
            if inspect.iscoroutinefunction(self.preprocess)
            else self.preprocess(body, headers)
        )

        return await to_thread.run_sync(self._predict_and_post, payload, headers)

    def _predict_and_post(
        self,
        payload: Any,
        headers: Optional[Dict[str, str]] = None,
    ) -> Any:
        self._predict_lock.acquire()
        defer_lock_release = False
        try:
            response = self.predict(payload, headers)
            response = self.postprocess(response, headers)
            if not isinstance(response, Generator):
                return response

            # Generator response
            if headers and headers.get("accept") == "application/json":
                return _convert_streamed_response_to_string(response)

            # Reaching here means streaming response, and need to defer releasing lock
            defer_lock_release = True
        finally:
            if not defer_lock_release:
                self._predict_lock.release()

        # Streaming response
        response_queue: Queue = Queue()

        def queue_response_chunks():
            # In a background thread, write the response chunks to a queue.
            # In the main thread, read data from the queue until a "None"
            # is written. This allows to us to use the predict lock only
            # around the actual predict, and does not create a dependency
            # on the client reading the entire response before releasing
            # the lock.
            try:
                for chunk in response:
                    response_queue.put(ResponseChunk(chunk))
                response_queue.put(None)
            finally:
                self._predict_lock.release()

        response_generate_thread = Thread(target=queue_response_chunks)
        response_generate_thread.start()
        return _response_generator(response_queue)


class ResponseChunk:
    def __init__(self, value):
        self.value = value


def _response_generator(queue: Queue):
    """
    When returning the stream result, simply read from the response queue until a `None`
    is reached.
    """
    while True:
        chunk = queue.get(timeout=STREAMING_RESPONSE_QUEUE_READ_TIMEOUT_SECS)
        if chunk is None:
            return
        else:
            yield chunk.value


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
