import asyncio
import dataclasses
import enum
import importlib
import importlib.util
import inspect
import logging
import os
import pathlib
import sys
import time
import weakref
from contextlib import asynccontextmanager
from enum import Enum
from functools import cached_property
from multiprocessing import Lock
from pathlib import Path
from threading import Thread
from typing import (
    Any,
    Callable,
    Dict,
    Optional,
    Tuple,
    Union,
)

import opentelemetry.sdk.trace as sdk_trace
import starlette.requests
import starlette.responses
from anyio import Semaphore, to_thread
from common import errors, tracing
from common.patches import apply_patches
from common.retry import retry
from common.schema import TrussSchema
from opentelemetry import trace
from pydantic import BaseModel
from shared import serialization
from shared.lazy_data_resolver import LazyDataResolver
from shared.secrets_resolver import SecretsResolver

if sys.version_info >= (3, 9):
    from typing import AsyncGenerator, Generator
else:
    from typing_extensions import AsyncGenerator, Generator

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


_ArgsType = Union[
    Tuple[Any],
    Tuple[Any, starlette.requests.Request],
    Tuple[starlette.requests.Request],
]


class _Sentinel:
    def __repr__(self) -> str:
        return "<Sentinel End of Queue>"


SENTINEL = _Sentinel()


def _is_request_type(obj: Any) -> bool:
    # issubclass raises an error (instead of returning False) if `obj` is not a type.
    try:
        return issubclass(obj, starlette.requests.Request)
    except Exception:
        return False


class ArgConfig(enum.Enum):
    INPUTS_ONLY = enum.auto()
    REQUEST_ONLY = enum.auto()
    INPUTS_AND_REQUEST = enum.auto()

    @classmethod
    def from_signature(
        cls,
        signature: inspect.Signature,
        method_name: str,
    ) -> "ArgConfig":
        parameters = list(signature.parameters.values())

        if len(parameters) == 1:
            if _is_request_type(parameters[0].annotation):
                return cls.REQUEST_ONLY
            return cls.INPUTS_ONLY

        elif len(parameters) == 2:
            # First arg can be whatever, except request. Second arg must be request.
            param1, param2 = parameters
            if param1.annotation:
                if _is_request_type(param1.annotation):
                    raise errors.ModelDefinitionError(
                        f"`{method_name}` method with two arguments is not allowed to "
                        "have request as first argument, request must be second. "
                        f"Got: {signature}"
                    )
            if not (param2.annotation and _is_request_type(param2.annotation)):
                raise errors.ModelDefinitionError(
                    f"`{method_name}` method with two arguments must have request as "
                    f"second argument (type annotated). Got: {signature} "
                )
            return cls.INPUTS_AND_REQUEST
        else:
            raise errors.ModelDefinitionError(
                f"`{method_name}` method cannot have more than two arguments. "
                f"Got: {signature}"
            )

    @classmethod
    def prepare_args(
        cls,
        descriptor: "MethodDescriptor",
        inputs: Any,
        request: starlette.requests.Request,
    ) -> _ArgsType:
        args: _ArgsType
        if descriptor.arg_config == ArgConfig.INPUTS_ONLY:
            args = (inputs,)
        elif descriptor.arg_config == ArgConfig.REQUEST_ONLY:
            args = (request,)
        elif descriptor.arg_config == ArgConfig.INPUTS_AND_REQUEST:
            args = (inputs, request)
        else:
            raise NotImplementedError(f"Arg config {descriptor.arg_config}.")
        return args


@dataclasses.dataclass
class MethodDescriptor:
    is_async: bool
    is_generator: bool
    arg_config: ArgConfig

    @classmethod
    def from_method(cls, method, method_name: str) -> "MethodDescriptor":
        return cls(
            is_async=inspect.iscoroutinefunction(method)
            or inspect.isasyncgenfunction(method),
            is_generator=inspect.isgeneratorfunction(method)
            or inspect.isasyncgenfunction(method),
            arg_config=ArgConfig.from_signature(inspect.signature(method), method_name),
        )


@dataclasses.dataclass
class ModelDescriptor:
    preprocess: Optional[MethodDescriptor]
    predict: MethodDescriptor
    postprocess: Optional[MethodDescriptor]
    truss_schema: Optional[TrussSchema]

    @cached_property
    def skip_input_parsing(self) -> bool:
        return self.predict.arg_config == ArgConfig.REQUEST_ONLY and (
            not self.preprocess or self.preprocess.arg_config == ArgConfig.REQUEST_ONLY
        )

    @classmethod
    def from_model(cls, model) -> "ModelDescriptor":
        if hasattr(model, "preprocess"):
            preprocess = MethodDescriptor.from_method(
                model.preprocess, method_name="preprocess"
            )
        else:
            preprocess = None

        if hasattr(model, "predict"):
            predict = MethodDescriptor.from_method(
                model.predict,
                method_name="predict",
            )
            if preprocess and predict.arg_config == ArgConfig.REQUEST_ONLY:
                raise errors.ModelDefinitionError(
                    "When using preprocessing, the predict method cannot only have the "
                    "request argument (because the result of preprocessing would be "
                    "discarded)."
                )
        else:
            raise errors.ModelDefinitionError(
                "Truss model must have a `predict` method."
            )

        if hasattr(model, "postprocess"):
            postprocess = MethodDescriptor.from_method(model.postprocess, "postprocess")
            if postprocess and postprocess.arg_config == ArgConfig.REQUEST_ONLY:
                raise errors.ModelDefinitionError(
                    "The postprocessing method cannot only have the request "
                    "argument (because the result of predict would be discarded)."
                )
        else:
            postprocess = None

        if preprocess:
            parameters = inspect.signature(model.preprocess).parameters
        else:
            parameters = inspect.signature(model.predict).parameters

        if postprocess:
            return_annotation = inspect.signature(model.postprocess).return_annotation
        else:
            return_annotation = inspect.signature(model.predict).return_annotation

        return cls(
            preprocess=preprocess,
            predict=predict,
            postprocess=postprocess,
            truss_schema=TrussSchema.from_signature(parameters, return_annotation),
        )


class ModelWrapper:
    _config: Dict
    _tracer: sdk_trace.Tracer
    _maybe_model: Optional[Any]
    _maybe_model_descriptor: Optional[ModelDescriptor]
    _logger: logging.Logger
    _status: "ModelWrapper.Status"
    _predict_semaphore: Semaphore

    class Status(Enum):
        NOT_READY = 0
        LOADING = 1
        READY = 2
        FAILED = 3

    def __init__(self, config: Dict, tracer: sdk_trace.Tracer):
        self._config = config
        self._tracer = tracer
        self._maybe_model = None
        self._maybe_model_descriptor = None
        self._logger = logging.getLogger()
        self.name = MODEL_BASENAME
        self._load_lock = Lock()
        self._status = ModelWrapper.Status.NOT_READY
        self._predict_semaphore = Semaphore(
            self._config.get("runtime", {}).get(
                "predict_concurrency", DEFAULT_PREDICT_CONCURRENCY
            )
        )

    @property
    def _model(self) -> Any:
        if self._maybe_model:
            return self._maybe_model
        else:
            raise errors.ModelNotReady(self.name)

    @property
    def model_descriptor(self) -> ModelDescriptor:
        if self._maybe_model_descriptor:
            return self._maybe_model_descriptor
        else:
            raise errors.ModelNotReady(self.name)

    @property
    def load_failed(self) -> bool:
        return self._status == ModelWrapper.Status.FAILED

    @property
    def ready(self) -> bool:
        return self._status == ModelWrapper.Status.READY

    @property
    def _model_file_name(self) -> str:
        return self._config["model_class_filename"]

    def start_load_thread(self):
        # Don't retry failed loads.
        if self._status == ModelWrapper.Status.NOT_READY:
            thread = Thread(target=self.load)
            thread.start()

    def load(self) -> bool:
        if self.ready:
            return True

        # if we are already loading, block on aquiring the lock;
        # this worker will return 503 while the worker with the lock is loading
        with self._load_lock:
            self._status = ModelWrapper.Status.LOADING
            self._logger.info("Executing model.load()...")
            try:
                start_time = time.perf_counter()
                self._load_impl()
                self._status = ModelWrapper.Status.READY
                self._logger.info(
                    f"Completed model.load() execution in {_elapsed_ms(start_time)} ms"
                )
                return True
            except Exception:
                self._logger.exception("Exception while loading model")
                self._status = ModelWrapper.Status.FAILED

        return False

    def _load_impl(self):
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
            self._logger.info("Loading truss model from file.")
            module_path = pathlib.Path(model_class_file_path).resolve()
            module_name = module_path.stem  # Use the file's name as the module name
            if not os.path.isfile(module_path):
                raise ImportError(
                    f"`{module_path}` is not a file. You must point to a python file where "
                    "the entrypoint chainlet is defined."
                )
            import_error_msg = f"Could not import `{module_path}`. Check path."
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            if not spec:
                raise ImportError(import_error_msg)
            if not spec.loader:
                raise ImportError(import_error_msg)
            module = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(module)
            except ImportError as e:
                if "attempted relative import" in str(e):
                    raise ImportError(
                        f"During import of `{model_class_file_path}`. "
                        f"Since Truss v0.9.36 relative imports (starting with '.') in "
                        "the top-level model file are no longer supported. Please "
                        "replace them with absolute imports. For guidance on importing "
                        "custom packages refer to our documentation "
                        "https://docs.baseten.co/truss-reference/config#packages"
                    ) from e

                raise

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
            self._maybe_model = model_class(**model_init_params)

        elif TRT_LLM_EXTENSION_NAME in extensions:
            self._logger.info("Loading TRT LLM extension as model.")
            # trt_llm extension allows model.py to be absent. It supplies its
            # own model class in that case.
            trt_llm_extension = extensions["trt_llm"]
            self._maybe_model = trt_llm_extension.model_override()
        else:
            raise RuntimeError("No module class file found")

        self._maybe_model_descriptor = ModelDescriptor.from_model(self._model)

        if hasattr(self._model, "load"):
            retry(
                self._model.load,
                NUM_LOAD_RETRIES,
                self._logger.warning,
                "Failed to load model.",
                gap_seconds=1.0,
            )

    async def preprocess(
        self,
        inputs: serialization.InputType,
        request: starlette.requests.Request,
    ) -> Any:
        descriptor = self.model_descriptor.preprocess
        if not descriptor:
            return inputs

        args = ArgConfig.prepare_args(descriptor, inputs, request)
        with errors.intercept_exceptions(self._logger, self._model_file_name):
            if descriptor.is_async:
                return await self._model.preprocess(*args)
            else:
                return await to_thread.run_sync(self._model.preprocess, *args)

    async def predict(
        self,
        inputs: Any,
        request: starlette.requests.Request,
    ) -> Union[serialization.OutputType, Any]:
        # The result can be a serializable data structure, byte-generator, a request,
        # or, if `postprocessing` is used, anything. In the last case postprocessing
        # must convert the result to something serializable.
        descriptor = self.model_descriptor.predict
        args = ArgConfig.prepare_args(descriptor, inputs, request)
        with errors.intercept_exceptions(self._logger, self._model_file_name):
            if descriptor.is_generator:
                # Even for async generators, don't await here.
                return self._model.predict(*args)
            if descriptor.is_async:
                return await self._model.predict(*args)
            # Offload sync functions to thread, to not block event loop.
            return await to_thread.run_sync(self._model.predict, *args)

    async def postprocess(
        self,
        result: Union[serialization.InputType, Any],
        request: starlette.requests.Request,
    ) -> serialization.OutputType:
        # The postprocess function can handle outputs of `predict`, but not
        # generators and responses - in that case predict must return directly
        # and postprocess is skipped.
        # The result type can be the same as for predict.
        descriptor = self.model_descriptor.postprocess
        if not descriptor:
            return result

        args = ArgConfig.prepare_args(descriptor, result, request)
        with errors.intercept_exceptions(self._logger, self._model_file_name):
            if descriptor.is_async:
                return await self._model.postprocess(*args)
            # Offload sync functions to thread, to not block event loop.
            return await to_thread.run_sync(self._model.postprocess, *args)

    async def _write_response_to_queue(
        self,
        queue: asyncio.Queue,
        generator: AsyncGenerator[bytes, None],
        span: trace.Span,
    ) -> None:
        with tracing.section_as_event(span, "write_response_to_queue"):
            try:
                async for chunk in generator:
                    await queue.put(chunk)
            except Exception as e:
                self._logger.exception(
                    "Exception while generating streamed response: " + str(e),
                    exc_info=errors.filter_traceback(self._model_file_name),
                )
            finally:
                await queue.put(SENTINEL)

    async def _stream_with_background_task(
        self,
        generator: Union[Generator[bytes, None, None], AsyncGenerator[bytes, None]],
        span: trace.Span,
        trace_ctx: trace.Context,
        release_and_end: Callable[[], None],
    ) -> AsyncGenerator[bytes, None]:
        # The streaming read timeout is the amount of time in between streamed chunk
        # before a timeout is triggered.
        streaming_read_timeout = self._config.get("runtime", {}).get(
            "streaming_read_timeout", STREAMING_RESPONSE_QUEUE_READ_TIMEOUT_SECS
        )
        async_generator = _force_async_generator(generator)
        # To ensure that a partial read from a client does not keep  the semaphore
        # claimed, we write all the data from the stream to the queue as it is produced,
        # irrespective of how fast it is consumed.
        # We then return a new generator that reads from the queue, and then
        # exits the semaphore block.
        response_queue: asyncio.Queue = asyncio.Queue()

        # `write_response_to_queue` keeps running the background until completion.
        gen_task = asyncio.create_task(
            self._write_response_to_queue(response_queue, async_generator, span)
        )
        # Defer the release of the semaphore until the write_response_to_queue task.
        gen_task.add_done_callback(lambda _: release_and_end())

        # The gap between responses in a stream must be < streaming_read_timeout
        # TODO: this whole buffering might be superfluous and sufficiently done by
        #   by the FastAPI server already. See `test_limit_concurrency_with_sse`.
        async def _buffered_response_generator() -> AsyncGenerator[bytes, None]:
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
                    if chunk == SENTINEL:
                        return
                    yield chunk

        return _buffered_response_generator()

    async def __call__(
        self,
        inputs: Optional[serialization.InputType],
        request: starlette.requests.Request,
    ) -> serialization.OutputType:
        """
        Returns result from: preprocess -> predictor -> postprocess.
        """
        with self._tracer.start_as_current_span("call-pre") as span_pre:
            with tracing.section_as_event(
                span_pre, "preprocess"
            ), tracing.detach_context():
                preprocess_result = await self.preprocess(inputs, request)

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
                predict_result = await self.predict(preprocess_result, request)

            if inspect.isgenerator(predict_result) or inspect.isasyncgen(
                predict_result
            ):
                if self.model_descriptor.postprocess:
                    with errors.intercept_exceptions(
                        self._logger, self._model_file_name
                    ):
                        raise errors.ModelDefinitionError(
                            "If the predict function returns a generator (streaming), "
                            "you cannot use postprocessing. Include all processing in "
                            "the predict method."
                        )

                if request.headers.get("accept") == "application/json":
                    # In the case of a streaming response, consume stream
                    # if the http accept header is set, and json is requested.
                    return await _gather_generator(predict_result)
                else:
                    return await self._stream_with_background_task(
                        predict_result,
                        span_predict,
                        detached_ctx,
                        release_and_end=get_defer_fn(),
                    )

            if isinstance(predict_result, starlette.responses.Response):
                if self.model_descriptor.postprocess:
                    with errors.intercept_exceptions(
                        self._logger, self._model_file_name
                    ):
                        raise errors.ModelDefinitionError(
                            "If the predict function returns a response object, "
                            "you cannot use postprocessing."
                        )
                if isinstance(predict_result, starlette.responses.StreamingResponse):
                    # Defer the semaphore release, using a weakref on the response.
                    # This might keep the semaphore longer than using "native" truss
                    # streaming, because here the criterion is not the production of
                    # data by the generator, but the span of handling the request by
                    # the fastAPI server.
                    weakref.finalize(predict_result, get_defer_fn())

                return predict_result

        with self._tracer.start_as_current_span("call-post") as span_post:
            with tracing.section_as_event(
                span_post, "postprocess"
            ), tracing.detach_context():
                postprocess_result = await self.postprocess(predict_result, request)

            if isinstance(postprocess_result, BaseModel):
                # If we return a pydantic object, convert it back to a dict
                with tracing.section_as_event(span_post, "dump-pydantic"):
                    final_result = postprocess_result.dict()
            else:
                final_result = postprocess_result
            return final_result


async def _gather_generator(
    predict_result: Union[AsyncGenerator[bytes, None], Generator[bytes, None, None]],
) -> str:
    return "".join(
        [str(chunk) async for chunk in _force_async_generator(predict_result)]
    )


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
        while True:
            # Note that this is the equivalent of running:
            # next(gen, FINAL_GENERATOR_VALUE) on a separate thread,
            # ensuring that if there is anything blocking in the generator,
            # it does not block the main loop.
            chunk = await to_thread.run_sync(next, gen, SENTINEL)
            if chunk == SENTINEL:
                return
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
