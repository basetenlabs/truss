import asyncio
import dataclasses
import enum
import importlib
import importlib.util
import inspect
import json
import logging
import os
import pathlib
import sys
import time
import weakref
from contextlib import asynccontextmanager
from functools import cached_property
from multiprocessing import Lock
from pathlib import Path
from threading import Thread
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple, Union, cast

import opentelemetry.sdk.trace as sdk_trace
import pydantic
import starlette.requests
import starlette.responses
from anyio import Semaphore, to_thread
from common import errors, tracing
from common.patches import apply_patches
from common.retry import retry
from common.schema import TrussSchema
from fastapi import HTTPException, WebSocket
from opentelemetry import trace
from shared import dynamic_config_resolver, serialization
from shared.lazy_data_resolver import LazyDataResolverV2
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
POLL_FOR_ENVIRONMENT_UPDATES_TIMEOUT_SECS = 30


class MethodName(str, enum.Enum):
    def _generate_next_value_(  # type: ignore[override]
        name: str, start: int, count: int, last_values: List[str]
    ) -> str:
        return name.lower()

    CHAT_COMPLETIONS = enum.auto()
    COMPLETIONS = enum.auto()
    IS_HEALTHY = enum.auto()
    POSTPROCESS = enum.auto()
    PREDICT = enum.auto()
    PREPROCESS = enum.auto()
    SETUP_ENVIRONMENT = enum.auto()
    WEBSOCKET = enum.auto()

    def __str__(self) -> str:
        return self.value


InputType = Union[serialization.JSONType, serialization.MsgPackType, pydantic.BaseModel]
OutputType = Union[
    serialization.JSONType,
    serialization.MsgPackType,
    Generator[bytes, None, None],
    AsyncGenerator[bytes, None],
    "starlette.responses.Response",
    pydantic.BaseModel,
]
ModelFn = Callable[..., Union[OutputType, Awaitable[OutputType]]]


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
    Tuple[()],
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


async def raise_if_disconnected(
    request: starlette.requests.Request, step_name: str
) -> None:
    if await request.is_disconnected():
        error_message = f"Client disconnected, skipping `{step_name}`."
        logging.warning(error_message)
        raise HTTPException(status_code=499, detail=error_message)


class ArgConfig(enum.Enum):
    NONE = enum.auto()
    INPUTS_ONLY = enum.auto()
    REQUEST_ONLY = enum.auto()
    INPUTS_AND_REQUEST = enum.auto()

    @classmethod
    def from_method(cls, method: Any, method_name: MethodName) -> "ArgConfig":
        signature = inspect.signature(method)
        parameters = list(signature.parameters.values())
        if len(parameters) == 0:
            return cls.NONE
        elif len(parameters) == 1:
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
        inputs: Any,
        request: starlette.requests.Request,
        descriptor: "MethodDescriptor",
    ) -> _ArgsType:
        args: _ArgsType
        if descriptor.arg_config == ArgConfig.NONE:
            args = ()
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
    method_name: MethodName
    method: ModelFn

    @classmethod
    def from_method(cls, method: Any, method_name: MethodName) -> "MethodDescriptor":
        return cls(
            is_async=cls._is_async(method),
            is_generator=cls._is_generator(method),
            arg_config=ArgConfig.from_method(method, method_name),
            method_name=method_name,
            # ArgConfig ensures that the Callable has an appropriate signature.
            method=cast(ModelFn, method),
        )

    @classmethod
    def _is_async(cls, method: Any):
        # We intentionally do not check inspect.isasyncgenfunction(method) because you cannot
        # `await` an async generator, you must use `async for` syntax.
        return inspect.iscoroutinefunction(method)

    @classmethod
    def _is_generator(cls, method: Any):
        return inspect.isgeneratorfunction(method) or inspect.isasyncgenfunction(method)


@dataclasses.dataclass
class ModelDescriptor:
    preprocess: Optional[MethodDescriptor]
    predict: Optional[MethodDescriptor]  # Websocket may replace predict.
    postprocess: Optional[MethodDescriptor]
    truss_schema: Optional[TrussSchema]
    setup_environment: Optional[MethodDescriptor]
    is_healthy: Optional[MethodDescriptor]
    completions: Optional[MethodDescriptor]
    chat_completions: Optional[MethodDescriptor]
    websocket: Optional[MethodDescriptor]

    @cached_property
    def skip_input_parsing(self) -> bool:
        return bool(
            self.predict
            and self.predict.arg_config == ArgConfig.REQUEST_ONLY
            and (
                not self.preprocess
                or self.preprocess.arg_config == ArgConfig.REQUEST_ONLY
            )
        )

    @classmethod
    def _gen_truss_schema(
        cls,
        predict: MethodDescriptor,
        preprocess: Optional[MethodDescriptor],
        postprocess: Optional[MethodDescriptor],
    ) -> TrussSchema:
        if preprocess:
            parameters = inspect.signature(preprocess.method).parameters
        else:
            parameters = inspect.signature(predict.method).parameters

        if postprocess:
            return_annotation = inspect.signature(postprocess.method).return_annotation
        else:
            return_annotation = inspect.signature(predict.method).return_annotation

        return TrussSchema.from_signature(parameters, return_annotation)

    @classmethod
    def _safe_extract_descriptor(
        cls, model_cls: Any, method_name: MethodName
    ) -> Union[MethodDescriptor, None]:
        if hasattr(model_cls, method_name):
            return MethodDescriptor.from_method(
                method=getattr(model_cls, method_name), method_name=method_name
            )
        return None

    @classmethod
    def from_model(cls, model_cls) -> "ModelDescriptor":
        setup = cls._safe_extract_descriptor(model_cls, MethodName.SETUP_ENVIRONMENT)
        completions = cls._safe_extract_descriptor(model_cls, MethodName.COMPLETIONS)
        chats = cls._safe_extract_descriptor(model_cls, MethodName.CHAT_COMPLETIONS)
        is_healthy = cls._safe_extract_descriptor(model_cls, MethodName.IS_HEALTHY)
        if is_healthy and is_healthy.arg_config != ArgConfig.NONE:
            raise errors.ModelDefinitionError(
                f"`{MethodName.IS_HEALTHY}` must have only one argument: `self`."
            )
        websocket = cls._safe_extract_descriptor(model_cls, MethodName.WEBSOCKET)
        predict = cls._safe_extract_descriptor(model_cls, MethodName.PREDICT)
        truss_schema, preprocess, postprocess = None, None, None

        preprocess = cls._safe_extract_descriptor(model_cls, MethodName.PREPROCESS)
        postprocess = cls._safe_extract_descriptor(model_cls, MethodName.POSTPROCESS)

        if websocket and (predict or preprocess or postprocess):
            raise errors.ModelDefinitionError(
                f"Truss model cannot have both `{MethodName.WEBSOCKET}` and any of "
                f"`{MethodName.PREDICT}`, `{MethodName.PREPROCESS}`, or "
                f"`{MethodName.POSTPROCESS} methods."
            )

        elif not (websocket or predict):
            raise errors.ModelDefinitionError(
                f"Truss model must have a `{MethodName.PREDICT}` or `{MethodName.WEBSOCKET}` method."
            )

        elif websocket:
            assert predict is None
            if websocket.arg_config != ArgConfig.INPUTS_ONLY:
                raise errors.ModelDefinitionError(
                    f"`{MethodName.WEBSOCKET}` must have only one argument: `websocket`."
                )
            if not websocket.is_async:
                raise errors.ModelDefinitionError(
                    f"`{MethodName.WEBSOCKET}` endpoints must be async function definitions."
                )
        elif predict:
            assert websocket is None
            if preprocess and predict.arg_config == ArgConfig.REQUEST_ONLY:
                raise errors.ModelDefinitionError(
                    f"When using `{MethodName.PREPROCESS}`, the {MethodName.PREDICT} method "
                    f"cannot only have the request argument (because the result of "
                    f"`{MethodName.PREPROCESS}` would be  discarded)."
                )

            if postprocess and postprocess.arg_config == ArgConfig.REQUEST_ONLY:
                raise errors.ModelDefinitionError(
                    f"The `{MethodName.POSTPROCESS}` method cannot only have the request "
                    f"argument (because the result of `{MethodName.PREDICT}` would be discarded)."
                )

            truss_schema = cls._gen_truss_schema(
                predict=predict, preprocess=preprocess, postprocess=postprocess
            )
        else:
            # This case should never happen, since above conditions should
            # be exhaustive.
            raise errors.ModelDefinitionError(
                "Unsupported method combination on truss model."
            )

        return cls(
            preprocess=preprocess,
            predict=predict,
            postprocess=postprocess,
            truss_schema=truss_schema,
            setup_environment=setup,
            is_healthy=is_healthy,
            completions=completions,
            chat_completions=chats,
            websocket=websocket,
        )


class ModelWrapper:
    _config: Dict
    _tracer: sdk_trace.Tracer
    _maybe_model: Optional[Any]
    _maybe_model_descriptor: Optional[ModelDescriptor]
    _logger: logging.Logger
    _status: "ModelWrapper.Status"
    _predict_semaphore: Semaphore
    _poll_for_environment_updates_task: Optional[asyncio.Task]
    _environment: Optional[dict]

    class Status(enum.Enum):
        NOT_READY = 0
        LOADING = 1
        READY = 2
        FAILED = 3

    def __init__(self, config: Dict, tracer: sdk_trace.Tracer):
        self._config = config
        self._tracer = tracer
        self._maybe_model = None
        self._maybe_model_descriptor = None
        # We need a logger that has all our server JSON logging setup applied in its
        # handlers and where this also hold in the loading thread. Creating a new
        # instance does not carry over the setup into the thread and using unspecified
        # `getLogger` may return non-compliant loggers if dependencies override the root
        # logger (c.g. https://github.com/numpy/numpy/issues/24213). We chose to get
        # the uvicorn logger that is set up in `truss_server`.
        self._logger = logging.getLogger("uvicorn")
        self.name = MODEL_BASENAME
        self._load_lock = Lock()
        self._status = ModelWrapper.Status.NOT_READY
        self._predict_semaphore = Semaphore(
            self._config.get("runtime", {}).get(
                "predict_concurrency", DEFAULT_PREDICT_CONCURRENCY
            )
        )
        self._poll_for_environment_updates_task = None
        self._environment = None

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
    def model_file_name(self) -> str:
        return self._config["model_class_filename"]

    @property
    def skip_input_parsing(self) -> bool:
        return self.model_descriptor.skip_input_parsing

    @property
    def truss_schema(self) -> Optional[TrussSchema]:
        return self.model_descriptor.truss_schema

    def start_load_thread(self):
        # Don't retry failed loads.
        if self._status == ModelWrapper.Status.NOT_READY:
            thread = Thread(target=self.load)
            thread.start()

    def load(self):
        if self.ready:
            return
        # if we are already loading, block on acquiring the lock;
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
            except Exception:
                self._logger.exception("Exception while loading model")
                self._status = ModelWrapper.Status.FAILED

    def _load_impl(self):
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)

        if "bundled_packages_dir" in self._config:
            bundled_packages_path = Path("/packages")
            if bundled_packages_path.exists():
                sys.path.append(str(bundled_packages_path))

        secrets = SecretsResolver.get_secrets(self._config)
        lazy_data_resolver = LazyDataResolverV2(data_dir)

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
                model_class, self._config, data_dir, secrets, lazy_data_resolver
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

        if self._maybe_model_descriptor.setup_environment:
            self._initialize_environment_before_load()
        if hasattr(self._model, "load"):
            retry(
                self._model.load,
                NUM_LOAD_RETRIES,
                self._logger.warning,
                "Failed to load model.",
                gap_seconds=1.0,
            )
        lazy_data_resolver.raise_if_not_collected()

    def setup_polling_for_environment_updates(self):
        self._poll_for_environment_updates_task = asyncio.create_task(
            self.poll_for_environment_updates()
        )

    def _initialize_environment_before_load(self):
        environment_str = dynamic_config_resolver.get_dynamic_config_value_sync(
            dynamic_config_resolver.ENVIRONMENT_DYNAMIC_CONFIG_KEY
        )
        if environment_str:
            environment_json = json.loads(environment_str)
            self._logger.info(
                f"Executing model.setup_environment with environment: {environment_json}"
            )
            # TODO: Support calling an async setup_environment() here once we support async load()
            self._model.setup_environment(environment_json)
            self._environment = environment_json

    async def setup_environment(self, environment: Optional[dict]) -> None:
        descriptor = self.model_descriptor.setup_environment
        if not descriptor:
            return
        self._logger.info(
            f"Executing model.setup_environment with environment: {environment}"
        )
        if descriptor.is_async:
            await self._model.setup_environment(environment)
        else:
            await to_thread.run_sync(self._model.setup_environment, environment)

    async def poll_for_environment_updates(self) -> None:
        last_modified_time = None
        environment_config_filename = (
            dynamic_config_resolver.get_dynamic_config_file_path(
                dynamic_config_resolver.ENVIRONMENT_DYNAMIC_CONFIG_KEY
            )
        )

        while True:
            # Give control back to the event loop while waiting for environment updates
            await asyncio.sleep(POLL_FOR_ENVIRONMENT_UPDATES_TIMEOUT_SECS)

            # Wait for load to finish before checking for environment updates
            if not self.ready:
                continue

            # Skip polling if no setup_environment implementation provided
            if not self.model_descriptor.setup_environment:
                break

            if environment_config_filename.exists():
                try:
                    current_mtime = os.path.getmtime(environment_config_filename)
                    if not last_modified_time or last_modified_time != current_mtime:
                        environment_str = await dynamic_config_resolver.get_dynamic_config_value_async(
                            dynamic_config_resolver.ENVIRONMENT_DYNAMIC_CONFIG_KEY
                        )
                        if environment_str:
                            last_modified_time = current_mtime
                            environment_json = json.loads(environment_str)
                            # Avoid rerunning `setup_environment` with the same environment
                            if self._environment != environment_json:
                                await self.setup_environment(environment_json)
                                self._environment = environment_json
                except Exception as e:
                    self._logger.exception(
                        f"Exception while setting up environment: {str(e)}",
                        exc_info=errors.filter_traceback(self.model_file_name),
                    )

    async def is_healthy(self) -> Optional[bool]:
        descriptor = self.model_descriptor.is_healthy
        is_healthy: Optional[bool] = None
        if not descriptor or self.load_failed:
            # return early with None if model does not have is_healthy method or load failed
            return None
        try:
            if descriptor.is_async:
                is_healthy = await self._model.is_healthy()
            else:
                # Offload sync functions to thread, to not block event loop.
                is_healthy = await to_thread.run_sync(self._model.is_healthy)
        except Exception as e:
            is_healthy = False
            self._logger.exception(
                f"Exception while checking if model is healthy: {str(e)}",
                exc_info=errors.filter_traceback(self.model_file_name),
            )
        if not is_healthy and self.ready:
            # self.ready evaluates to True when the model's load function has completed,
            # we will only log health check failures to model logs when the model's load has completed
            self._logger.warning("Health check failed.")
        return is_healthy

    async def preprocess(
        self, inputs: InputType, request: starlette.requests.Request
    ) -> Any:
        descriptor = self.model_descriptor.preprocess
        assert descriptor, (
            f"`{MethodName.PREPROCESS}` must only be called if model has it."
        )
        return await self._execute_user_model_fn(inputs, request, descriptor)

    async def _predict(
        self, inputs: Any, request: starlette.requests.Request
    ) -> Union[OutputType, Any]:
        # The result can be a serializable data structure, byte-generator, a request,
        # or, if `postprocessing` is used, anything. In the last case postprocessing
        # must convert the result to something serializable.
        descriptor = self.model_descriptor.predict
        assert descriptor, (
            f"`{MethodName.PREDICT}` must only be called if model has it."
        )
        return await self._execute_user_model_fn(inputs, request, descriptor)

    async def postprocess(
        self, result: Union[InputType, Any], request: starlette.requests.Request
    ) -> OutputType:
        # The postprocess function can handle outputs of `predict`, but not
        # generators and responses - in that case predict must return directly
        # and postprocess is skipped.
        # The result type can be the same as for predict.
        descriptor = self.model_descriptor.postprocess
        assert descriptor, (
            f"`{MethodName.POSTPROCESS}` must only be called if model has it."
        )
        return await self._execute_user_model_fn(result, request, descriptor)

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
                    f"Exception while generating streamed response: {str(e)}",
                    exc_info=errors.filter_traceback(self.model_file_name),
                )
            finally:
                await queue.put(SENTINEL)

    async def _stream_with_background_task(
        self,
        generator: Union[Generator[bytes, None, None], AsyncGenerator[bytes, None]],
        span: trace.Span,
        trace_ctx: trace.Context,
        cleanup_fn: Callable[[], None],
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
        gen_task.add_done_callback(lambda _: cleanup_fn())

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
                        response_queue.get(), timeout=streaming_read_timeout
                    )
                    if chunk == SENTINEL:
                        return
                    yield chunk

        return _buffered_response_generator()

    async def _execute_user_model_fn(
        self,
        inputs: Union[InputType, Any],
        request: starlette.requests.Request,
        descriptor: MethodDescriptor,
    ) -> OutputType:
        await raise_if_disconnected(request, descriptor.method_name)
        args = ArgConfig.prepare_args(inputs, request, descriptor)
        with errors.intercept_exceptions(self._logger, self.model_file_name):
            if descriptor.is_generator:
                # Even for async generators, don't await here.
                return descriptor.method(*args)
            if descriptor.is_async:
                return await cast(Awaitable[OutputType], descriptor.method(*args))
            return await to_thread.run_sync(descriptor.method, *args)

    async def _execute_model_endpoint(
        self,
        inputs: InputType,
        request: starlette.requests.Request,
        descriptor: MethodDescriptor,
    ) -> OutputType:
        """
        Wraps the execution of any model code other than `predict`.
        """
        await raise_if_disconnected(request, descriptor.method_name)
        fn_span = self._tracer.start_span(f"call-{descriptor.method_name}")
        with tracing.section_as_event(
            fn_span, descriptor.method_name, detach=True
        ) as detached_ctx:
            result = await self._execute_user_model_fn(inputs, request, descriptor)

        if inspect.isgenerator(result) or inspect.isasyncgen(result):
            return await self._handle_generator_response(
                request, result, fn_span, detached_ctx
            )

        return result

    def _should_gather_generator(self, request: starlette.requests.Request) -> bool:
        # The OpenAI SDK sends an accept header for JSON even in a streaming context,
        # but we need to stream results back for client compatibility. Luckily,
        # we can differentiate by looking at the user agent (e.g. OpenAI/Python 1.61.0)
        user_agent = request.headers.get("user-agent", "")
        if "openai" in user_agent.lower():
            return False
        # TODO(nikhil): determine if we can safely deprecate this behavior.
        return request.headers.get("accept") == "application/json"

    async def _handle_generator_response(
        self,
        request: starlette.requests.Request,
        generator: Union[Generator[bytes, None, None], AsyncGenerator[bytes, None]],
        span: trace.Span,
        trace_ctx: trace.Context,
        get_cleanup_fn: Callable[[], Callable[[], None]] = lambda: lambda: None,
    ):
        if self._should_gather_generator(request):
            return await _gather_generator(generator)
        else:
            return await self._stream_with_background_task(
                generator, span, trace_ctx, cleanup_fn=get_cleanup_fn()
            )

    def _get_descriptor_or_raise(
        self, descriptor: Optional[MethodDescriptor], method_name: MethodName
    ) -> MethodDescriptor:
        if not descriptor:
            raise errors.ModelMethodNotImplemented(
                f"`{method_name}` must only be called if model has it."
            )

        return descriptor

    async def predict(
        self, inputs: Optional[InputType], request: starlette.requests.Request
    ) -> OutputType:
        """
        Returns result from: preprocess -> predictor -> postprocess.
        """
        if self.model_descriptor.preprocess:
            with self._tracer.start_as_current_span("call-pre") as span_pre:
                with tracing.section_as_event(span_pre, "preprocess", detach=True):
                    preprocess_result = await self.preprocess(inputs, request)
        else:
            preprocess_result = inputs

        span_predict = self._tracer.start_span("call-predict")
        async with deferred_semaphore_and_span(
            self._predict_semaphore, span_predict
        ) as get_defer_fn:
            with tracing.section_as_event(
                span_predict, "predict", detach=True
            ) as detached_ctx:
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
                predict_result = await self._predict(preprocess_result, request)

            if inspect.isgenerator(predict_result) or inspect.isasyncgen(
                predict_result
            ):
                if self.model_descriptor.postprocess:
                    with errors.intercept_exceptions(
                        self._logger, self.model_file_name
                    ):
                        raise errors.ModelDefinitionError(
                            "If the predict function returns a generator (streaming), "
                            "you cannot use postprocessing. Include all processing in "
                            "the predict method."
                        )

                return await self._handle_generator_response(
                    request,
                    predict_result,
                    span_predict,
                    detached_ctx,
                    get_cleanup_fn=get_defer_fn,
                )

            if isinstance(predict_result, starlette.responses.Response):
                if self.model_descriptor.postprocess:
                    with errors.intercept_exceptions(
                        self._logger, self.model_file_name
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

        if self.model_descriptor.postprocess:
            with self._tracer.start_as_current_span("call-post") as span_post:
                with tracing.section_as_event(span_post, "postprocess", detach=True):
                    postprocess_result = await self.postprocess(predict_result, request)
                return postprocess_result
        else:
            return predict_result

    async def completions(
        self, inputs: InputType, request: starlette.requests.Request
    ) -> OutputType:
        descriptor = self._get_descriptor_or_raise(
            self.model_descriptor.completions, MethodName.COMPLETIONS
        )
        return await self._execute_model_endpoint(inputs, request, descriptor)

    async def chat_completions(
        self, inputs: InputType, request: starlette.requests.Request
    ) -> OutputType:
        descriptor = self._get_descriptor_or_raise(
            self.model_descriptor.chat_completions, MethodName.CHAT_COMPLETIONS
        )
        return await self._execute_model_endpoint(inputs, request, descriptor)

    async def websocket(self, ws: WebSocket) -> None:
        descriptor = self.model_descriptor.websocket
        assert descriptor, "websocket can only be invoked if present on model."
        assert descriptor.is_async, "websocket endpoints are enforced to be async."
        await self._model.websocket(ws)


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
                    extension_name, config, data_dir, secrets, lazy_data_resolver
                )
                extensions[extension_name] = extension
    return extensions


def _init_extension(extension_name: str, config, data_dir, secrets, lazy_data_resolver):
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
        model_init_params["lazy_data_resolver"] = lazy_data_resolver
    if _signature_accepts_keyword_arg(signature, "environment"):
        environment = None
        environment_str = dynamic_config_resolver.get_dynamic_config_value_sync(
            dynamic_config_resolver.ENVIRONMENT_DYNAMIC_CONFIG_KEY
        )
        if environment_str:
            environment = json.loads(environment_str)
        model_init_params["environment"] = environment
    return model_init_params
