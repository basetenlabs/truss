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
            return self._model.predict(payload)  # type: ignore
        except Exception:
            response = {}
            logging.exception("Exception while running predict")
            response["error"] = {"traceback": traceback.format_exc()}
            return response

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
            # In the case of a streaming response:
            #    1. If the user passes an accept header of "application/json",
            #       simply consume the full response and return it as a string.
            #    2. If the user does not, take the response generator, and in a separate
            #       thread, write the response chunks to a queue. In the main thread, read
            #       data from the queue until a "None" is written. This allows to us to use the
            #       predict lock only around the actually predict, and does not create a dependency
            #       on the client reading the entire response before releasing the lock.
            if headers and headers.get("accept") == "application/json":
                response = _convert_streamed_response_to_string(response)
            else:
                response_queue: Queue = Queue()

                response_generate_thread = Thread(
                    target=_queue_response,
                    args=(response, response_queue, self._predict_lock),
                )
                response_generate_thread.start()

                return _response_generator(response_queue)

        return response


class ResponseChunk:
    def __init__(self, value):
        self.value = value


def _queue_response(response_generator: Generator, queue: Queue, lock: Lock):
    """
    When the predict function returns a Generator (in the case of streaming), simply
    write all of the contents in a queue. When we return the result, it will read from
    this queue.

    We write the data using the ResponseChunk class so that we can communicate more easily
    when the response is complete.
    """
    with lock:
        for chunk in response_generator:
            queue.put(ResponseChunk(chunk))
        queue.put(None)


def _response_generator(queue: Queue):
    """
    When returning the stream result, simply read from the response queue until a `None`
    is reached.
    """
    while True:
        chunk = queue.get()
        if chunk is None:
            return
        else:
            yield chunk.value


class DeferLockToGenerator:
    """
    Context manager that accepts a lock, and wraps a block of code with that lock.
    It provides the ability to defer the lock to the end of a generator,
    if the code chooses.

    If you defer the lock release, the generator MUST be read, otherwise there is
    a risk of the lock never being released.
    """

    def __init__(self, lock: Lock):
        self.lock = lock
        self.deferred_to_generator = False

    def __enter__(self):
        self.lock.acquire()
        return self

    def __call__(self, generator: Generator):
        if self.deferred_to_generator:
            raise RuntimeError("Cannot defer to multiple generators in single block.")

        def inner():
            try:
                for chunk in generator:
                    yield chunk
            finally:
                self.lock.release()

        self.deferred_to_generator = True
        return inner()

    def __exit__(self, exc_type, exc_value, exc_tb):
        if not self.deferred_to_generator:
            self.lock.release()


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
