import asyncio
import importlib
import inspect
import logging
import os
import sys
import time
from collections.abc import Generator
from contextlib import asynccontextmanager
from enum import Enum
from multiprocessing import Lock
from pathlib import Path
from threading import Thread
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Coroutine,
    Dict,
    NoReturn,
    Optional,
    Set,
    TypeVar,
    Union,
)

import pydantic
from anyio import Semaphore, to_thread
from common.patches import apply_patches
from common.retry import retry
from common.schema import TrussSchema
from fastapi import HTTPException
from pydantic import BaseModel
from shared.lazy_data_resolver import LazyDataResolver
from shared.secrets_resolver import SecretsResolver
from typing_extensions import ParamSpec

MODEL_BASENAME = "model"

NUM_LOAD_RETRIES = int(os.environ.get("NUM_LOAD_RETRIES_TRUSS", "1"))
STREAMING_RESPONSE_QUEUE_READ_TIMEOUT_SECS = 60
DEFAULT_PREDICT_CONCURRENCY = 1


class DeferredSemaphoreManager:
    """
    Helper class for supported deferred semaphore release.
    """

    def __init__(self, semaphore: Semaphore):
        self.semaphore = semaphore
        self.deferred = False

    def defer(self):
        """
        Track that this semaphore is to be deferred, and return
        a release method that the context block can use to release
        the semaphore.
        """
        self.deferred = True

        return self.semaphore.release


@asynccontextmanager
async def deferred_semaphore(semaphore: Semaphore):
    """
    Context manager that allows deferring the release of a semaphore.
    It yields a DeferredSemaphoreManager -- in your use of this context manager,
    if you call DeferredSemaphoreManager.defer(), you will get back a function that releases
    the semaphore that you must call.
    """
    semaphore_manager = DeferredSemaphoreManager(semaphore)
    await semaphore.acquire()

    try:
        yield semaphore_manager
    finally:
        if not semaphore_manager.deferred:
            semaphore.release()


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
        self._status = ModelWrapper.Status.NOT_READY
        self._predict_semaphore = Semaphore(
            self._config.get("runtime", {}).get(
                "predict_concurrency", DEFAULT_PREDICT_CONCURRENCY
            )
        )
        self._background_tasks: Set[asyncio.Task] = set()
        self.truss_schema: TrussSchema = None

    def load(self) -> bool:
        if self.ready:
            return self.ready

        # if we are already loading, block on aquiring the lock;
        # this worker will return 503 while the worker with the lock is loading
        with self._load_lock:
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

        return self.ready

    def start_load(self):
        if self.should_load():
            thread = Thread(target=self.load)
            thread.start()

    def load_failed(self) -> bool:
        return self._status == ModelWrapper.Status.FAILED

    def should_load(self) -> bool:
        # don't retry failed loads
        return not self._status == ModelWrapper.Status.FAILED and not self.ready

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
        if _signature_accepts_keyword_arg(model_class_signature, "lazy_data_resolver"):
            model_init_params["lazy_data_resolver"] = LazyDataResolver(data_dir).fetch()
        apply_patches(
            self._config.get("apply_library_patches", True),
            self._config["requirements"],
        )
        self._model = model_class(**model_init_params)

        self.set_truss_schema()

        if hasattr(self._model, "load"):
            retry(
                self._model.load,
                NUM_LOAD_RETRIES,
                self._logger.warn,
                "Failed to load model.",
                gap_seconds=1.0,
            )

    def set_truss_schema(self):
        parameters = (
            inspect.signature(self._model.preprocess).parameters
            if hasattr(self._model, "preprocess")
            else inspect.signature(self._model.predict).parameters
        )

        outputs_annotation = (
            inspect.signature(self._model.postprocess).return_annotation
            if hasattr(self._model, "postprocess")
            else inspect.signature(self._model.predict).return_annotation
        )

        self.truss_schema = TrussSchema.from_signature(parameters, outputs_annotation)

    async def preprocess(
        self,
        payload: Any,
        headers: Optional[Dict[str, str]] = None,
    ) -> Any:
        if not hasattr(self._model, "preprocess"):
            return payload

        if inspect.iscoroutinefunction(self._model.preprocess):
            return await _intercept_exceptions_async(self._model.preprocess)(payload)
        else:
            return await to_thread.run_sync(
                _intercept_exceptions_sync(self._model.preprocess), payload
            )

    async def predict(
        self,
        payload: Any,
        headers: Optional[Dict[str, str]] = None,
    ) -> Any:
        # It's possible for the user's predict function to be a:
        #   1. Generator function (function that returns a generator)
        #   2. Async generator (function that returns async generator)
        # In these cases, just return the generator or async generator,
        # as we will be propagating these up. No need for await at this point.
        #   3. Coroutine -- in this case, await the predict function as it is async
        #   4. Normal function -- in this case, offload to a separate thread to prevent
        #      blocking the main event loop
        if inspect.isasyncgenfunction(
            self._model.predict
        ) or inspect.isgeneratorfunction(self._model.predict):
            return self._model.predict(payload)

        if inspect.iscoroutinefunction(self._model.predict):
            return await _intercept_exceptions_async(self._model.predict)(payload)

        return await to_thread.run_sync(
            _intercept_exceptions_sync(self._model.predict), payload
        )

    async def postprocess(
        self,
        response: Any,
        headers: Optional[Dict[str, str]] = None,
    ) -> Any:
        # Similar to the predict function, it is possible for postprocess
        # to return either a generator or async generator, in which case
        # just return the generator.
        #
        # It can also return a coroutine or just be a function, in which
        # case either await, or offload to a thread respectively.
        if not hasattr(self._model, "postprocess"):
            return response

        if inspect.isasyncgenfunction(
            self._model.postprocess
        ) or inspect.isgeneratorfunction(self._model.postprocess):
            return self._model.postprocess(response)

        if inspect.iscoroutinefunction(self._model.postprocess):
            return await _intercept_exceptions_async(self._model.postprocess)(response)

        return await to_thread.run_sync(
            _intercept_exceptions_sync(self._model.postprocess), response
        )

    async def write_response_to_queue(
        self, queue: asyncio.Queue, generator: AsyncGenerator
    ):
        try:
            async for chunk in generator:
                await queue.put(ResponseChunk(chunk))
        except Exception as e:
            self._logger.exception("Exception while reading stream response: " + str(e))
        finally:
            await queue.put(None)

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

        # The streaming read timeout is the amount of time in between streamed chunks before a timeout is triggered
        streaming_read_timeout = self._config.get("runtime", {}).get(
            "streaming_read_timeout", STREAMING_RESPONSE_QUEUE_READ_TIMEOUT_SECS
        )

        if self.truss_schema is not None:
            try:
                body = self.truss_schema.input_type(**body)
            except pydantic.ValidationError as e:
                self._logger.info("Request Validation Error")
                raise HTTPException(
                    status_code=400, detail=f"Request Validation Error, {str(e)}"
                ) from e

        payload = await self.preprocess(body, headers)

        async with deferred_semaphore(self._predict_semaphore) as semaphore_manager:
            response = await self.predict(payload, headers)

            # Streaming cases
            if inspect.isgenerator(response) or inspect.isasyncgen(response):
                if hasattr(self._model, "postprocess"):
                    logging.warning(
                        "Predict returned a streaming response, while a postprocess is defined."
                        "Note that in this case, the postprocess will run within the predict lock."
                    )

                    response = await self.postprocess(response)

                async_generator = _force_async_generator(response)

                if headers and headers.get("accept") == "application/json":
                    # In the case of a streaming response, consume stream
                    # if the http accept header is set, and json is requested.
                    return await _convert_streamed_response_to_string(async_generator)

                # To ensure that a partial read from a client does not cause the semaphore
                # to stay claimed, we immediately write all of the data from the stream to a
                # queue. We then return a new generator that reads from the queue, and then
                # exit the semaphore block.
                response_queue: asyncio.Queue = asyncio.Queue()

                # This task will be triggered and run in the background.
                task = asyncio.create_task(
                    self.write_response_to_queue(response_queue, async_generator)
                )

                # We add the task to the ModelWrapper instance to ensure it does
                # not get garbage collected after the predict method completes,
                # and continues running.
                self._background_tasks.add(task)

                # Defer the release of the semaphore until the write_response_to_queue
                # task.
                semaphore_release_function = semaphore_manager.defer()
                task.add_done_callback(lambda _: semaphore_release_function())
                task.add_done_callback(self._background_tasks.discard)

                # The gap between responses in a stream must be < streaming_read_timeout
                async def _response_generator():
                    while True:
                        chunk = await asyncio.wait_for(
                            response_queue.get(),
                            timeout=streaming_read_timeout,
                        )
                        if chunk is None:
                            return
                        yield chunk.value

                return _response_generator()

        processed_response = await self.postprocess(response)

        if isinstance(processed_response, BaseModel):
            # If we return a pydantic object, convert it back to a dict
            processed_response = processed_response.dict()
        return processed_response


class ResponseChunk:
    def __init__(self, value):
        self.value = value


async def _convert_streamed_response_to_string(response: AsyncGenerator):
    return "".join([str(chunk) async for chunk in response])


def _force_async_generator(gen: Union[Generator, AsyncGenerator]) -> AsyncGenerator:
    """
    Takes a generator, and converts it into an async generator if it is not already.
    """
    if inspect.isasyncgen(gen):
        return gen

    async def _convert_generator_to_async():
        """
        Runs each iteration of the generator in an offloaded thread, to ensure
        the main loop is not blocked, and yield to create an async generator.
        """
        FINAL_GENERATOR_VALUE = object()
        while True:
            # Note that this is the equivalent of running:
            # next(gen, FINAL_GENERATOR_VALUE) on a separate thread,
            # ensuring that if there is anything blocking in the generator,
            # it does not block the main loop.
            chunk = await to_thread.run_sync(next, gen, FINAL_GENERATOR_VALUE)
            if chunk == FINAL_GENERATOR_VALUE:
                break
            yield chunk

    return _convert_generator_to_async()


def _signature_accepts_keyword_arg(signature: inspect.Signature, kwarg: str) -> bool:
    return kwarg in signature.parameters or _signature_accepts_kwargs(signature)


def _signature_accepts_kwargs(signature: inspect.Signature) -> bool:
    for param in signature.parameters.values():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            return True
    return False


def _elapsed_ms(since_micro_seconds: float) -> int:
    return int((time.perf_counter() - since_micro_seconds) * 1000)


def _handle_exception(exception: Exception) -> NoReturn:
    # Note that logger.exception logs the stacktrace, such that the user can
    # debug this error from the logs.
    if isinstance(exception, HTTPException):
        logging.exception("Model raised HTTPException")
        raise exception
    else:
        logging.exception("Internal Server Error")
        raise HTTPException(status_code=500, detail="Internal Server Error")


_P = ParamSpec("_P")
_R = TypeVar("_R")


def _intercept_exceptions_sync(func: Callable[_P, _R]) -> Callable[_P, _R]:
    def inner(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            _handle_exception(e)

    return inner


def _intercept_exceptions_async(
    func: Callable[_P, Coroutine[Any, Any, _R]],
) -> Callable[_P, Coroutine[Any, Any, _R]]:
    async def inner(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            _handle_exception(e)

    return inner
