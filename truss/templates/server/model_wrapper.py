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
    Mapping,
    NoReturn,
    Optional,
    TypeVar,
    Union,
)

import opentelemetry.sdk.trace as sdk_trace
import pydantic
from anyio import Semaphore, to_thread
from common import tracing
from common.patches import apply_patches
from common.retry import retry
from common.schema import TrussSchema
from fastapi import HTTPException
from opentelemetry import trace
from pydantic import BaseModel
from shared.lazy_data_resolver import LazyDataResolver
from shared.secrets_resolver import SecretsResolver
from typing_extensions import ParamSpec

MODEL_BASENAME = "model"

NUM_LOAD_RETRIES = int(os.environ.get("NUM_LOAD_RETRIES_TRUSS", "1"))
STREAMING_RESPONSE_QUEUE_READ_TIMEOUT_SECS = 60
DEFAULT_PREDICT_CONCURRENCY = 1
EXTENSIONS_DIR_NAME = "extensions"
EXTENSION_CLASS_NAME = "Extension"
EXTENSION_FILE_NAME = "extension"
TRT_LLM_EXTENSION_NAME = "trt_llm"


@asynccontextmanager
async def deferred_semaphore_and_span(
    semaphore: Semaphore, span: trace.Span
) -> AsyncGenerator[Callable[[], Callable[[], None]], None]:
    """
    Context manager that allows deferring the release of a semaphore and the ending of a
    trace span.

    Yields a function that, when called, releases the semaphore and ends the span.
    If that function is not called, the resources are cleand up when exiting.
    """
    await semaphore.acquire()
    trace.use_span(span, end_on_exit=False)
    deferred = False

    def release_and_end() -> None:
        semaphore.release()
        span.end()

    def defer() -> Callable[[], None]:
        nonlocal deferred
        deferred = True
        return release_and_end

    try:
        yield defer
    finally:
        if not deferred:
            release_and_end()


class ModelWrapper:
    _tracer: sdk_trace.Tracer

    class Status(Enum):
        NOT_READY = 0
        LOADING = 1
        READY = 2
        FAILED = 3

    def __init__(self, config: Dict, tracer: sdk_trace.Tracer):
        self._config = config
        self._tracer = tracer
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

        secrets = SecretsResolver.get_secrets(self._config)
        lazy_data_resolver = LazyDataResolver(data_dir)

        apply_patches(
            self._config.get("apply_library_patches", True),
            self._config["requirements"],
        )

        extensions = _init_extensions(
            self._config, data_dir, secrets, lazy_data_resolver
        )
        for extension in extensions.values():
            extension.load()

        model_class_file_path = (
            Path(self._config["model_module_dir"])
            / self._config["model_class_filename"]
        )
        if model_class_file_path.exists():
            model_module_path = Path(self._config["model_class_filename"])
            model_module_name = str(model_module_path.with_suffix(""))
            module = importlib.import_module(
                f"{self._config['model_module_dir']}.{model_module_name}"
            )
            model_class = getattr(module, self._config["model_class_name"])
            model_init_params = _prepare_init_args(
                model_class,
                self._config,
                data_dir,
                secrets,
                lazy_data_resolver,
            )
            signature = inspect.signature(model_class)
            for ext_name, ext in extensions.items():
                if _signature_accepts_keyword_arg(signature, ext_name):
                    model_init_params[ext_name] = ext.model_args()
            self._model = model_class(**model_init_params)
        elif TRT_LLM_EXTENSION_NAME in extensions:
            # trt_llm extension allows model.py to be absent. It supplies its
            # own model class in that case.
            trt_llm_extension = extensions["trt_llm"]
            self._model = trt_llm_extension.model_override()
        else:
            raise RuntimeError("No module class file found")

        self.set_truss_schema()

        if hasattr(self._model, "load"):
            retry(
                self._model.load,
                NUM_LOAD_RETRIES,
                self._logger.warning,
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
        self, queue: asyncio.Queue, generator: AsyncGenerator, span: trace.Span
    ):
        with tracing.section_as_event(span, "write_response_to_queue"):
            try:
                async for chunk in generator:
                    # TODO: consider checking `request.is_disconnected()` for
                    #   client-side cancellations and freeing resources.
                    await queue.put(ResponseChunk(chunk))
            except Exception as e:
                self._logger.exception(
                    "Exception while reading stream response: " + str(e)
                )
            finally:
                await queue.put(None)

    async def _streaming_post_process(self, response: Any, span: trace.Span) -> Any:
        if hasattr(self._model, "postprocess"):
            logging.warning(
                "Predict returned a streaming response, while a postprocess is defined."
                "Note that in this case, the postprocess will run within the predict lock."
            )
            with tracing.section_as_event(
                span, "postprocess"
            ), tracing.detach_context():
                response = await self.postprocess(response)

        return response

    async def _gather_generator(self, response: Any, span: trace.Span) -> str:
        # In the case of gathering, it might make more sense to apply the postprocess
        # to the gathered result, but that would be inconsistent with streaming.
        # In general, it might even be better to strictly forbid postprocessing
        # for generators.
        response = await self._streaming_post_process(response, span)
        return await _convert_streamed_response_to_string(
            _force_async_generator(response)
        )

    async def _stream_with_background_task(
        self,
        response: Any,
        span: trace.Span,
        trace_ctx: trace.Context,
        release_and_end: Callable[[], None],
    ):
        # The streaming read timeout is the amount of time in between streamed chunk
        # before a timeout is triggered.
        streaming_read_timeout = self._config.get("runtime", {}).get(
            "streaming_read_timeout", STREAMING_RESPONSE_QUEUE_READ_TIMEOUT_SECS
        )
        response = await self._streaming_post_process(response, span)
        async_generator = _force_async_generator(response)
        # To ensure that a partial read from a client does not keep  the semaphore
        # claimed, we write all the data from the stream to the queue as it is produced,
        # irrespective of how fast it is consumed.
        # We then return a new generator that reads from the queue, and then
        # exits the semaphore block.
        response_queue: asyncio.Queue = asyncio.Queue()

        # `write_response_to_queue` keeps running the background until completion.
        gen_task = asyncio.create_task(
            self.write_response_to_queue(response_queue, async_generator, span)
        )
        # Defer the release of the semaphore until the write_response_to_queue task.
        gen_task.add_done_callback(lambda _: release_and_end())

        # The gap between responses in a stream must be < streaming_read_timeout
        async def _buffered_response_generator():
            # `span` is tied to the "producer" `gen_task` which might complete before
            # "consume" part here finishes, therefore a dedicated span is required.
            # Because all of this code is inside a `detach_context` block, we
            # explicitly propagate the tracing context for this span.
            with self._tracer.start_as_current_span(
                "buffered-response-generator", context=trace_ctx
            ):
                while True:
                    chunk = await asyncio.wait_for(
                        response_queue.get(),
                        timeout=streaming_read_timeout,
                    )
                    if chunk is None:
                        return
                    yield chunk.value

        return _buffered_response_generator()

    async def __call__(
        self, body: Any, headers: Optional[Mapping[str, str]] = None
    ) -> Union[Dict, Generator, AsyncGenerator, str]:
        """Method to call predictor or explainer with the given input.

        Args:
            body (Any): Request payload body.
            headers (Dict): Request headers.

        Returns:
            Dict: Response output from preprocess -> predictor -> postprocess
            Generator: In case of streaming response
            String: in case of non-streamed generator (the string is the JSON result).
        """
        with self._tracer.start_as_current_span("call-pre") as span_pre:
            if self.truss_schema is not None:
                try:
                    with tracing.section_as_event(span_pre, "parse-pydantic"):
                        body = self.truss_schema.input_type(**body)
                except pydantic.ValidationError as e:
                    self._logger.info("Request Validation Error")
                    raise HTTPException(
                        status_code=400, detail=f"Request Validation Error, {str(e)}"
                    ) from e
            with tracing.section_as_event(
                span_pre, "preprocess"
            ), tracing.detach_context():
                payload = await self.preprocess(body)

        span_predict = self._tracer.start_span("call-predict")
        async with deferred_semaphore_and_span(
            self._predict_semaphore, span_predict
        ) as get_defer_fn:
            with tracing.section_as_event(
                span_predict, "predict"
            ), tracing.detach_context() as detached_ctx:
                # To prevent span pollution, we need to make sure spans created by user
                # code don't inherit context from our spans (which happens even if
                # different tracer instances are used).
                # Therefor, predict is run in `detach_context`.
                # There is one caveat with streaming predictions though:
                # The context manager only detaches spans that are created outside
                # the generator loop that yields the stream (because the parts of the
                # loop body will be executed in a "deferred" way (same reasoning as for
                # using `deferred_semaphore_and_span`). We assume that here that
                # creating spans inside the loop body is very unlikely. In order to
                # exactly handle that case we would need to apply `detach_context`
                # around each `next`-invocation that consumes the generator, which is
                # prohibitive.
                response = await self.predict(payload)

            if inspect.isgenerator(response) or inspect.isasyncgen(response):
                if headers and headers.get("accept") == "application/json":
                    # In the case of a streaming response, consume stream
                    # if the http accept header is set, and json is requested.
                    return await self._gather_generator(response, span_predict)
                else:
                    return await self._stream_with_background_task(
                        response,
                        span_predict,
                        detached_ctx,
                        release_and_end=get_defer_fn(),
                    )

        with self._tracer.start_as_current_span("call-post") as span_post:
            with tracing.section_as_event(
                span_post, "postprocess"
            ), tracing.detach_context():
                processed_response = await self.postprocess(response)

            if isinstance(processed_response, BaseModel):
                # If we return a pydantic object, convert it back to a dict
                with tracing.section_as_event(span_post, "dump-pydantic"):
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


def _init_extensions(config, data_dir, secrets, lazy_data_resolver):
    extensions = {}
    extensions_path = Path(__file__).parent / EXTENSIONS_DIR_NAME
    if extensions_path.exists():
        for extension_path in extensions_path.iterdir():
            if extension_path.is_dir():
                extension_name = extension_path.name
                extension = _init_extension(
                    extension_name,
                    config,
                    data_dir,
                    secrets,
                    lazy_data_resolver,
                )
                extensions[extension_name] = extension
    return extensions


def _init_extension(
    extension_name: str,
    config,
    data_dir,
    secrets,
    lazy_data_resolver,
):
    extension_module = importlib.import_module(
        f"{EXTENSIONS_DIR_NAME}.{extension_name}.{EXTENSION_FILE_NAME}"
    )
    extension_class = getattr(extension_module, EXTENSION_CLASS_NAME)
    init_args = _prepare_init_args(
        extension_class,
        config=config,
        data_dir=data_dir,
        secrets=secrets,
        lazy_data_resolver=lazy_data_resolver,
    )
    return extension_class(**init_args)


def _prepare_init_args(klass, config, data_dir, secrets, lazy_data_resolver):
    """Prepares init params based on signature.

    Used to pass params to extension and model class' __init__ function.
    """
    signature = inspect.signature(klass)
    model_init_params = {}
    if _signature_accepts_keyword_arg(signature, "config"):
        model_init_params["config"] = config
    if _signature_accepts_keyword_arg(signature, "data_dir"):
        model_init_params["data_dir"] = data_dir
    if _signature_accepts_keyword_arg(signature, "secrets"):
        model_init_params["secrets"] = secrets
    if _signature_accepts_keyword_arg(signature, "lazy_data_resolver"):
        model_init_params["lazy_data_resolver"] = lazy_data_resolver.fetch()
    return model_init_params
