import abc
import ast
import atexit
import collections
import contextlib
import contextvars
import enum
import functools
import importlib.util
import inspect
import logging
import os
import pathlib
import pprint
import re
import sys
import types
import warnings
from importlib.abc import Loader
from typing import (
    Any,
    AsyncIterator,
    Callable,
    ClassVar,
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    Optional,
    Protocol,
    Type,
    TypeVar,
    Union,
    final,
    get_args,
    get_origin,
)

import pydantic
from typing_extensions import ParamSpec

from truss.base import custom_types, trt_llm_config
from truss_chains import private_types, public_types, utils

_SIMPLE_TYPES = {int, float, complex, bool, str, bytes, None, pydantic.BaseModel}
_SIMPLE_CONTAINERS = {list, dict}
_STREAM_TYPES = {str, bytes}

_DOCS_URL_CHAINING = (
    "https://docs.baseten.co/chains/concepts#depends-call-other-chainlets"
)
_DOCS_URL_LOCAL = "https://docs.baseten.co/chains/guide#local-development"
_DOCS_URL_STREAMING = "https://docs.baseten.co/chains/guide#streaming"

# A "neutral dummy" endpoint descriptor if validation fails, this allows to safely
# continue checking for more errors.
_DUMMY_ENDPOINT_DESCRIPTOR = private_types.EndpointAPIDescriptor(
    input_args=[], output_types=[], is_async=False, is_streaming=False
)

_DISPLAY_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]+$")

ChainletT = TypeVar("ChainletT", bound=private_types.ABCChainlet)
_P = ParamSpec("_P")
_R = TypeVar("_R")


# Error Collector ######################################################################


class _ErrorKind(str, enum.Enum):
    TYPE_ERROR = enum.auto()
    IO_TYPE_ERROR = enum.auto()
    MISSING_API_ERROR = enum.auto()
    INVALID_CONFIG_ERROR = enum.auto()


class _ErrorLocation(custom_types.SafeModel):
    src_path: str
    line: Optional[int] = None
    chainlet_name: Optional[str] = None
    method_name: Optional[str] = None

    def __str__(self) -> str:
        value = f"{self.src_path}:{self.line}"
        if self.chainlet_name and self.method_name:
            value = f"{value} ({self.chainlet_name}.{self.method_name})"
        elif self.chainlet_name:
            value = f"{value} ({self.chainlet_name})"
        else:
            assert not self.chainlet_name
        return value


class _ValidationError(custom_types.SafeModel):
    msg: str
    kind: _ErrorKind
    location: _ErrorLocation

    def __str__(self) -> str:
        return f"{self.location} [kind: {self.kind.name}]: {self.msg}"


class _ErrorCollector:
    _errors: list[_ValidationError]

    def __init__(self) -> None:
        self._errors = []
        # This hook is for the case of just running the Chainlet file, without
        # making a push - we want to surface the errors at exit.
        atexit.register(self.maybe_display_errors)

    def clear(self) -> None:
        self._errors.clear()

    def collect(self, error):
        self._errors.append(error)

    @property
    def has_errors(self) -> bool:
        return bool(self._errors)

    @property
    def num_errors(self) -> int:
        return len(self._errors)

    def format_errors(self) -> str:
        parts = []
        for error in self._errors:
            parts.append(str(error))

        return "\n".join(parts)

    def maybe_display_errors(self) -> None:
        if self.has_errors:
            sys.stderr.write(self.format_errors())
            sys.stderr.write("\n")


_global_error_collector = _ErrorCollector()


def _collect_error(msg: str, kind: _ErrorKind, location: _ErrorLocation):
    _global_error_collector.collect(
        _ValidationError(msg=msg, kind=kind, location=location)
    )


def raise_validation_errors() -> None:
    """Raises validation errors as combined ``ChainsUsageError``"""
    if _global_error_collector.has_errors:
        error_msg = _global_error_collector.format_errors()
        _global_error_collector.clear()  # Clear errors so `atexit` won't display them
        raise public_types.ChainsUsageError(
            "The user defined code does not comply with the required spec, "
            f"please fix below:\n{error_msg}"
        )


def raise_validation_errors_before(f: Callable[_P, _R]) -> Callable[_P, _R]:
    """Raises validation errors as combined ``ChainsUsageError`` before invoking `f`."""

    @functools.wraps(f)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        raise_validation_errors()
        return f(*args, **kwargs)

    return wrapper


class _BaseProvisionMarker:
    """A marker for object to be dependency injected by the framework."""


class ContextDependencyMarker(_BaseProvisionMarker):
    def __str__(self) -> str:
        return f"{self.__class__.__name__}"

    def __getattr__(self, item: str) -> Any:
        logging.error(f"Attempting to access attribute `{item}` on `{self}`.")
        raise public_types.ChainsRuntimeError(
            "It seems `chains.depends_context()` was used, but not as an argument "
            "to the `__init__` method of a Chainlet - This is not supported."
            f"See {_DOCS_URL_CHAINING}.\n"
            "Example of correct `__init__` with context:\n"
            f"{_example_chainlet_code()}"
        )


class ChainletDependencyMarker(_BaseProvisionMarker):
    chainlet_cls: Type[private_types.ABCChainlet]
    retries: int

    def __init__(
        self,
        chainlet_cls: Type[private_types.ABCChainlet],
        options: public_types.RPCOptions,
    ) -> None:
        self.chainlet_cls = chainlet_cls
        self.options = options

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.chainlet_cls.name})"

    def __getattr__(self, item: str) -> Any:
        logging.error(f"Attempting to access attribute `{item}` on `{self}`.")
        raise public_types.ChainsRuntimeError(
            f"It seems `chains.depends({self.chainlet_cls.name})` was used, but "
            "not as an argument to the `__init__` method of a Chainlet - This is not "
            "supported. Dependency Chainlets must be passed as init arguments.\n"
            f"See {_DOCS_URL_CHAINING}.\n"
            "Example of correct `__init__` with dependencies:\n"
            f"{_example_chainlet_code()}"
        )


# Validation of Chainlet class definition ##############################################


@functools.cache
def _example_chainlet_code() -> str:
    # Note: this function requires all chains modules to be initialized, because
    # in `example_chainlet` the full chainlet validation process is triggered.
    # To avoid circular import dependencies, `_example_chainlet_code` should only be
    # called on erroneous code branches (which will not be triggered if
    # `example_chainlet` is free of errors).
    try:
        from truss_chains.reference_code import reference_chainlet
    # If `example_chainlet` fails validation and `_example_chainlet_code` is
    # called as a result of that, we have a circular import ("partially initialized
    # module 'truss_chains.example_chainlet' ...").
    except AttributeError:
        logging.error("`reference_chainlet` is broken.", exc_info=True, stack_info=True)
        return "<EXAMPLE CODE MISSING/BROKEN>"

    example_name = reference_chainlet.HelloWorld.name
    return _get_cls_source(reference_chainlet.__file__, example_name)


@functools.cache
def _example_model_code() -> str:
    try:
        from truss_chains.reference_code import reference_model
    except AttributeError:
        logging.error("`reference_model` is broken.", exc_info=True, stack_info=True)
        return "<EXAMPLE CODE MISSING/BROKEN>"

    example_name = reference_model.HelloWorld.name
    return _get_cls_source(reference_model.__file__, example_name)


def _get_cls_source(src_path: str, target_class_name: str) -> str:
    source = pathlib.Path(src_path).read_text()
    tree = ast.parse(source)
    class_code = ""
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == target_class_name:
            # Extract the source code of the class definition
            lines = source.splitlines()
            class_code = "\n".join(lines[node.lineno - 1 : node.end_lineno])
            break
    return class_code


def _instantiation_error_msg(cls_name: str, location: Optional[str] = None) -> str:
    location_format = f"{location}\n" if location else ""
    return (
        f"Error when instantiating Chainlet `{cls_name}`.\n"
        f"{location_format}"
        "Chainlets cannot be naively instantiated. Possible fixes:\n"
        "1. To use Chainlets as dependencies in other Chainlets ('chaining'), "
        f"add them as init argument. See {_DOCS_URL_CHAINING}.\n"
        f"2. For local / debug execution, use the `{run_local.__name__}`-"
        f"context. See {_DOCS_URL_LOCAL}. You cannot use helper functions to "
        "instantiate the Chain in this case.\n"
        "3. Push the chain and call the remote endpoint.\n"
        "Example of correct `__init__` with dependencies:\n"
        f"{_example_chainlet_code()}"
    )


def _validate_io_type(
    annotation: Any, param_name: str, location: _ErrorLocation
) -> None:
    """
    For Chainlet I/O (both data or parameters), we allow simple types
    (int, str, float...) and `list` or `dict` containers of these.
    Any deeper nested and structured data must be typed as a pydantic model.
    """
    containers_str = [c.__name__ for c in _SIMPLE_CONTAINERS]
    types_str = [c.__name__ if c is not None else "None" for c in _SIMPLE_TYPES]
    if isinstance(annotation, str):
        _collect_error(
            f"A string-valued type annotation was found for `{param_name}` of type "
            f"`{annotation}`. Use only actual types objects and avoid "
            "`from __future__ import annotations` (if needed upgrade python).",
            _ErrorKind.IO_TYPE_ERROR,
            location,
        )
        return
    if annotation in _SIMPLE_TYPES or annotation == public_types.WebSocketProtocol:
        return

    error_msg = (
        f"Unsupported I/O type for `{param_name}` of type `{annotation}`. "
        "Supported are:\n"
        f"\t* simple types: {types_str}\n"
        "\t* containers of these simple types, with annotated item types: "
        f"{containers_str}, e.g. `dict[str, int]` (use built-in types, not "
        "`typing.Dict`).\n"
        "\t* For complicated / nested data structures: `pydantic` models."
    )
    if isinstance(annotation, types.GenericAlias):
        if get_origin(annotation) not in _SIMPLE_CONTAINERS:
            _collect_error(error_msg, _ErrorKind.IO_TYPE_ERROR, location)
            return
        args = get_args(annotation)
        for arg in args:
            if not (
                arg in _SIMPLE_TYPES or utils.issubclass_safe(arg, pydantic.BaseModel)
            ):
                _collect_error(error_msg, _ErrorKind.IO_TYPE_ERROR, location)
                return
            pass
        return
    if utils.issubclass_safe(annotation, pydantic.BaseModel):
        return

    _collect_error(error_msg, _ErrorKind.IO_TYPE_ERROR, location)


def _validate_streaming_output_type(
    annotation: Any, location: _ErrorLocation
) -> private_types.StreamingTypeDescriptor:
    origin = get_origin(annotation)
    assert origin in (collections.abc.AsyncIterator, collections.abc.Iterator)
    args = get_args(annotation)
    if len(args) < 1:
        stream_types = sorted(list(x.__name__ for x in _STREAM_TYPES))
        _collect_error(
            f"Iterators must be annotated with type (one of {stream_types}).",
            _ErrorKind.IO_TYPE_ERROR,
            location,
        )
        return private_types.StreamingTypeDescriptor(
            raw=annotation, origin_type=origin, arg_type=bytes
        )

    assert len(args) == 1, "Iterator type annotations cannot have more than 1 arg."
    arg = args[0]
    if arg not in _STREAM_TYPES:
        msg = (
            "Streaming endpoints (containing `yield` statements) can only yield string "
            "or byte items. For streaming structured pydantic data, use `stream_writer`"
            "and `stream_reader` helpers.\n"
            f"See streaming docs: {_DOCS_URL_STREAMING}"
        )
        _collect_error(msg, _ErrorKind.IO_TYPE_ERROR, location)

    return private_types.StreamingTypeDescriptor(
        raw=annotation, origin_type=origin, arg_type=arg
    )


def _validate_method_signature(
    method_name: str, location: _ErrorLocation, params: list[inspect.Parameter]
) -> None:
    if len(params) == 0:
        _collect_error(
            f"`{method_name}` must be a method, i.e. with `{private_types.SELF_ARG_NAME}` as "
            "first argument. Got function with no arguments.",
            _ErrorKind.TYPE_ERROR,
            location,
        )
    elif params[0].name != private_types.SELF_ARG_NAME:
        _collect_error(
            f"`{method_name}` must be a method, i.e. with `{private_types.SELF_ARG_NAME}` as "
            f"first argument. Got `{params[0].name}` as first argument.",
            _ErrorKind.TYPE_ERROR,
            location,
        )


def _validate_endpoint_params(
    params: list[inspect.Parameter], location: _ErrorLocation
) -> list[private_types.InputArg]:
    _validate_method_signature(private_types.RUN_REMOTE_METHOD_NAME, location, params)
    input_args = []
    for param in params[1:]:  # Skip self argument.
        if param.annotation == inspect.Parameter.empty:
            _collect_error(
                "Arguments of endpoints must have type annotations. "
                f"Parameter `{param.name}` has no type annotation.",
                _ErrorKind.IO_TYPE_ERROR,
                location,
            )
        else:
            _validate_io_type(param.annotation, param.name, location)
            type_descriptor = private_types.TypeDescriptor(raw=param.annotation)
            is_optional = param.default != inspect.Parameter.empty
            input_args.append(
                private_types.InputArg(
                    name=param.name, type=type_descriptor, is_optional=is_optional
                )
            )
    return input_args


def _validate_endpoint_output_types(
    annotation: Any, signature, location: _ErrorLocation, is_streaming: bool
) -> list[private_types.TypeDescriptor]:
    has_streaming_type = False
    if annotation == inspect.Parameter.empty:
        _collect_error(
            "Return values of endpoints must be type annotated. Got:\n"
            f"\t{location.method_name}{signature} -> !MISSING!",
            _ErrorKind.IO_TYPE_ERROR,
            location,
        )
        return []
    origin = get_origin(annotation)
    if origin is tuple:
        output_types = []
        for i, arg in enumerate(get_args(annotation)):
            _validate_io_type(arg, f"return_type[{i}]", location)
            output_types.append(private_types.TypeDescriptor(raw=arg))

    elif origin in (collections.abc.AsyncIterator, collections.abc.Iterator):
        output_types = [_validate_streaming_output_type(annotation, location)]
        has_streaming_type = True
        if not is_streaming:
            _collect_error(
                "If the endpoint returns an iterator (streaming), it must have `yield` "
                "statements.",
                _ErrorKind.IO_TYPE_ERROR,
                location,
            )
    else:
        _validate_io_type(annotation, "return_type", location)
        output_types = [private_types.TypeDescriptor(raw=annotation)]

    if is_streaming and not has_streaming_type:
        _collect_error(
            "If the endpoint is streaming (has `yield` statements), the return type "
            "must be an iterator (e.g. `AsyncIterator[bytes]`). Got:\n"
            f"\t{location.method_name}{signature} -> {annotation}",
            _ErrorKind.IO_TYPE_ERROR,
            location,
        )

    return output_types


def _validate_websocket_endpoint(
    descriptor: private_types.EndpointAPIDescriptor, location: _ErrorLocation
) -> None:
    if any(arg.is_websocket for arg in descriptor.output_types):
        _collect_error(
            "Websockets cannot be used as output type.",
            _ErrorKind.IO_TYPE_ERROR,
            location,
        )

    if not any(arg.type.is_websocket for arg in descriptor.input_args):
        return

    if len(descriptor.input_args) > 1:
        _collect_error(
            "When using a websocket as input, no other arguments are allowed.",
            _ErrorKind.IO_TYPE_ERROR,
            location,
        )

    if len(descriptor.output_types) != 1 or descriptor.output_types[0].raw != None:  # noqa: E711
        _collect_error(
            "Websocket endpoints must have `None` as return type.",
            _ErrorKind.IO_TYPE_ERROR,
            location,
        )

    if descriptor.is_streaming:
        # Redundant since output type None is required.
        _collect_error(
            "Websocket endpoints cannot reurn itesators. Use the websocket itself to stream data.",
            _ErrorKind.IO_TYPE_ERROR,
            location,
        )

    if not descriptor.is_async:
        _collect_error(
            "Websocket endpoints must be async.", _ErrorKind.IO_TYPE_ERROR, location
        )


def _validate_and_describe_endpoint(
    cls: Type[private_types.ABCChainlet], location: _ErrorLocation
) -> private_types.EndpointAPIDescriptor:
    """The "endpoint method" of a Chainlet must have the following signature:

    ```
    [async] def run_remote(
        self, [param_0: anno_0, param_1: anno_1 = default_1, ...]) -> ret_anno:
    ```

    * The name must be `run_remote` for Chainlets, or `predict` for Models.
    * It can be sync or async def.
    * The number and names of parameters are arbitrary, both positional and named
      parameters are ok.
    * All parameters and the return value must have type annotations. See
      `_validate_io_type` for valid types.
    * Generators are allowed, too (but not yet supported).
    """
    if not hasattr(cls, cls.endpoint_method_name):
        _collect_error(
            f"{cls.entity_type}s must have a `{cls.endpoint_method_name}` method.",
            _ErrorKind.MISSING_API_ERROR,
            location,
        )
        return _DUMMY_ENDPOINT_DESCRIPTOR

    # This is the unbound method.
    endpoint_method = getattr(cls, cls.endpoint_method_name)
    line = inspect.getsourcelines(endpoint_method)[1]
    location = location.model_copy(
        update={"line": line, "method_name": cls.endpoint_method_name}
    )

    if not inspect.isfunction(endpoint_method):
        _collect_error("`Endpoints must be a method.", _ErrorKind.TYPE_ERROR, location)
        # If it's not a function, it might be a class var and subsequent inspections
        # fail.
        return _DUMMY_ENDPOINT_DESCRIPTOR
    signature = inspect.signature(endpoint_method)
    input_args = _validate_endpoint_params(
        list(signature.parameters.values()), location
    )

    if inspect.isasyncgenfunction(endpoint_method):
        is_async = True
        is_streaming = True
    elif inspect.iscoroutinefunction(endpoint_method):
        is_async = True
        is_streaming = False
    else:
        is_async = False
        is_streaming = inspect.isgeneratorfunction(endpoint_method)

    output_types = _validate_endpoint_output_types(
        signature.return_annotation, signature, location, is_streaming
    )

    if is_streaming:
        if not is_async:
            _collect_error(
                "`Streaming endpoints (containing `yield` statements) are only "
                "supported for async endpoints.",
                _ErrorKind.IO_TYPE_ERROR,
                location,
            )

    if not is_async:
        warnings.warn(
            f"`{cls.endpoint_method_name}` must be an async (coroutine) function in future releases. "
            "Replace `def run_remote(...)` with `async def run_remote(...)`. "
            "Local testing and execution can be done with  "
            "`asyncio.run(my_chainlet.run_remote(...))`.\n"
            "Note on concurrency: previously sync functions were run in threads by the "
            "Truss server.\n"
            "For some frameworks this was **unsafe** (e.g. in torch the CUDA context "
            "is not thread-safe).\n"
            "Additionally, python threads hold the GIL and therefore might not give "
            "actual throughput gains.\n"
            "To achieve safe and performant concurrency, use framework-specific async "
            "APIs (e.g. AsyncLLMEngine for vLLM) or generic async batching like such "
            "as https://github.com/hussein-awala/async-batcher.",
            DeprecationWarning,
            stacklevel=1,
        )
    descriptor = private_types.EndpointAPIDescriptor(
        name=cls.endpoint_method_name,
        input_args=input_args,
        output_types=output_types,
        is_async=is_async,
        is_streaming=is_streaming,
    )
    _validate_websocket_endpoint(descriptor, location)
    return descriptor


def _get_generic_class_type(var):
    """Extracts `SomeGeneric` from `SomeGeneric` or `SomeGeneric[T]` uniformly."""
    origin = get_origin(var)
    return origin if origin is not None else var


class _ChainletInitValidator:
    """The `__init__`-method of a Chainlet must have the following signature:

    ```
    def __init__(
        self,
        [dep_0: dep_0_type = truss_chains.depends(dep_0_class),]
        [dep_1: dep_1_type = truss_chains.depends(dep_1_class),]
        ...
        [dep_N: dep_N_type = truss_chains.provides(dep_N_class),]
        [context: truss_chains.Context = truss_chains.depends_context()]
    ) -> None:
    ```
    * The context argument is optionally trailing and must have a default constructed
     with the  `provide_context` directive.
    * The names and number of Chainlet "dependency" arguments are arbitrary.
    * Default values for dependencies must be constructed with the `depends` directive
      to make the dependency injection work. The argument to `depends` must be a
      Chainlet class.
    * The type annotation for dependencies can be a Chainlet class, but it can also be
      a `Protocol` with an equivalent `run` method (e.g. for getting correct type
      checks when providing fake Chainlets for local testing.). It may be omitted if
      the type is clear from the RHS.
    """

    _location: _ErrorLocation
    _cls: Type[private_types.ABCChainlet]
    has_context: bool = False
    validated_dependencies: Mapping[str, private_types.DependencyDescriptor] = {}

    def __init__(
        self, cls: Type[private_types.ABCChainlet], location: _ErrorLocation
    ) -> None:
        self._cls = cls
        if not self._cls.has_custom_init():
            self.has_context = False
            self.validated_dependencies = {}
            return
        line = inspect.getsourcelines(cls.__init__)[1]
        self._location = location.model_copy(
            update={"line": line, "method_name": "__init__"}
        )

        params = list(inspect.signature(cls.__init__).parameters.values())
        self._validate_args(params)

    def _validate_args(self, params: list[inspect.Parameter]):
        # Each validation pops of "processed" arguments from the list.
        self._validate_self_arg(params)
        self._validate_context_arg(params)
        self._validate_dependencies(params)

    def _validate_self_arg(self, params: list[inspect.Parameter]):
        if len(params) == 0:
            _collect_error(
                "Methods must have first argument `self`, got no arguments.",
                _ErrorKind.TYPE_ERROR,
                self._location,
            )
            return params
        param = params.pop(0)
        if param.name != private_types.SELF_ARG_NAME:
            _collect_error(
                f"Methods must have first argument `self`, got `{param.name}`.",
                _ErrorKind.TYPE_ERROR,
                self._location,
            )

    def _validate_context_arg(self, params: list[inspect.Parameter]):
        def make_context_error_msg():
            return (
                f"If `{self._cls.entity_type}` uses context for initialization, it "
                f"must have `{private_types.CONTEXT_ARG_NAME}` argument of type "
                f"`{public_types.DeploymentContext}` as the last argument.\n"
                f"Got arguments: `{params}`.\n"
                "Example of correct `__init__` with context:\n"
                f"{self._example_code()}"
            )

        if not params:
            return

        has_context = params[-1].name == private_types.CONTEXT_ARG_NAME
        has_context_marker = isinstance(params[-1].default, ContextDependencyMarker)
        if has_context ^ has_context_marker:
            _collect_error(
                make_context_error_msg(), _ErrorKind.TYPE_ERROR, self._location
            )

        if not has_context:
            return

        self.has_context = True
        param = params.pop(-1)
        param_type = _get_generic_class_type(param.annotation)
        # We are lenient and allow omitting the type annotation for context.
        if (
            (param_type is not None)
            and (param_type != inspect.Parameter.empty)
            and (not utils.issubclass_safe(param_type, public_types.DeploymentContext))
        ):
            _collect_error(
                make_context_error_msg(), _ErrorKind.TYPE_ERROR, self._location
            )
        if not isinstance(param.default, ContextDependencyMarker):
            _collect_error(
                f"Incorrect default value `{param.default}` for `context` argument. "
                "Example of correct `__init__` with dependencies:\n"
                f"{self._example_code()}",
                _ErrorKind.TYPE_ERROR,
                self._location,
            )

    def _validate_dependencies(self, params: list[inspect.Parameter]):
        used = set()
        dependencies = {}

        if params and not self._cls.supports_dependencies:
            _collect_error(
                f"The only supported argument to `__init__` for {self._cls.entity_type}s "
                f"is the optional context argument.",
                _ErrorKind.TYPE_ERROR,
                self._location,
            )
            return
        for param in params:
            marker = self._validate_dependency_param(param)
            if marker is None:
                continue
            if marker.chainlet_cls in used:
                _collect_error(
                    f"The same Chainlet class cannot be used multiple times for "
                    f"different arguments. Got previously used "
                    f"`{marker.chainlet_cls.__name__}` for `{param.name}`.",
                    _ErrorKind.TYPE_ERROR,
                    self._location,
                )

            if get_descriptor(marker.chainlet_cls).endpoint.is_websocket:
                _collect_error(
                    f"The dependency chainlet `{marker.chainlet_cls.__name__}` for "
                    f"`{param.name}` uses a websocket. But websockets can only be used "
                    "in the entrypoint, not in 'inner' chainlets.",
                    _ErrorKind.TYPE_ERROR,
                    self._location,
                )

            dependencies[param.name] = private_types.DependencyDescriptor(
                chainlet_cls=marker.chainlet_cls, options=marker.options
            )
            used.add(marker.chainlet_cls)

        self.validated_dependencies = dependencies

    def _validate_dependency_param(
        self, param: inspect.Parameter
    ) -> Optional[ChainletDependencyMarker]:
        """
        Returns a valid ChainletDependencyMarker if found, None otherwise.
        """
        # TODO: handle subclasses, unions, optionals, check default value etc.
        if param.name == private_types.CONTEXT_ARG_NAME:
            _collect_error(
                f"The init argument name `{private_types.CONTEXT_ARG_NAME}` is reserved for "
                "the optional context argument, which must be trailing if used. Example "
                "of correct `__init__` with context:\n"
                f"{self._example_code()}",
                _ErrorKind.TYPE_ERROR,
                self._location,
            )

        if not isinstance(param.default, ChainletDependencyMarker):
            _collect_error(
                f"Any arguments of a Chainlet's __init__ (besides `context`) must have "
                "dependency Chainlets with default values from `chains.depends`-directive. "
                f"Got `{param}`.\n"
                f"Example of correct `__init__` with dependencies:\n"
                f"{self._example_code()}",
                _ErrorKind.TYPE_ERROR,
                self._location,
            )
            return None

        chainlet_cls = param.default.chainlet_cls
        if not utils.issubclass_safe(chainlet_cls, private_types.ABCChainlet):
            _collect_error(
                f"`chains.depends` must be used with a Chainlet class as argument, got "
                f"{chainlet_cls} instead.",
                _ErrorKind.TYPE_ERROR,
                self._location,
            )
            return None
        # Check type annotation.
        # Also lenient with type annotation: since the RHS / default is asserted to be a
        # chainlet class, proper type inference is possible even without annotation.
        # TODO: `Protocol` is not a proper class and this might be version dependent.
        #   Find a better way to inspect this.
        if not (
            param.annotation == inspect.Parameter.empty
            or utils.issubclass_safe(param.annotation, Protocol)  # type: ignore[arg-type]
            or utils.issubclass_safe(chainlet_cls, param.annotation)
        ):
            _collect_error(
                f"The type annotation for `{param.name}` must be a class/subclass of the "
                "Chainlet type specified by `chains.provides` or a compatible "
                f"typing.Protocol`. Got `{param.annotation}`.",
                _ErrorKind.TYPE_ERROR,
                self._location,
            )
        return param.default  # The Marker.

    @functools.cache
    def _example_code(self) -> str:
        if self._cls.entity_type == private_types.EntityType.MODEL:
            return _example_model_code()
        return _example_chainlet_code()


def _validate_config_class_variable(
    cls: Type[private_types.ABCChainlet],
    location: _ErrorLocation,
    var_name: str,
    expected_type: Type[pydantic.BaseModel],
) -> None:
    if not isinstance(var_value := getattr(cls, var_name), expected_type):
        _collect_error(
            f"{cls.entity_type}s must have a `{var_name}` class variable of type "
            f"`{expected_type}`. Got `{type(var_value)}`.",
            _ErrorKind.TYPE_ERROR,
            location,
        )


def _validate_display_name(
    cls: Type[private_types.ABCChainlet], location: _ErrorLocation
) -> None:
    if cls.entity_type == private_types.EntityType.MODEL:
        return  # Model can have any name.

    if not bool(_DISPLAY_NAME_RE.match(cls.display_name)):
        _collect_error(
            f"Chainlet display name `{cls.display_name}` must match `{_DISPLAY_NAME_RE.pattern}` regex.",
            _ErrorKind.INVALID_CONFIG_ERROR,
            location,
        )


def _validate_engine_builder_fields(
    remote_config: public_types.RemoteConfig, location: _ErrorLocation
) -> None:
    # TODO: these checks are not very tightly coupled to `_gen_truss_config` - find
    #  a better way to do this.
    violations = []
    # `model_fields_set` does not work reliably here, so we compare against defaults.
    if remote_config.docker_image != utils.get_pydantic_field_default_value(
        public_types.RemoteConfig, "docker_image"
    ):
        violations.append("docker_image")
    if remote_config.assets.get_spec().cached != utils.get_pydantic_field_default_value(
        public_types.AssetSpec, "cached"
    ):
        violations.append("assets.cached")
    if (
        remote_config.assets.get_spec().external_data
        != utils.get_pydantic_field_default_value(
            public_types.AssetSpec, "external_data"
        )
    ):
        violations.append("assets.external_data")
    if remote_config.options.health_checks != utils.get_pydantic_field_default_value(
        public_types.ChainletOptions, "health_checks"
    ):
        violations.append("options.health_checks")
    if violations:
        _collect_error(
            f"{private_types.EntityType.ENGINE_BUILDER_MODEL}s don't support these "
            f"`remote_config` fields: {violations}. Leave them unset "
            f"(at their defaults) for this chainlet.",
            _ErrorKind.INVALID_CONFIG_ERROR,
            location,
        )


def _validate_health_check(
    cls: Type[private_types.ABCChainlet], location: _ErrorLocation
) -> Optional[private_types.HealthCheckAPIDescriptor]:
    """The `is_healthy` method of a Chainlet must have the following signature:
    ```
    [async] def is_healthy(self) -> bool:
    ```
    * The name must be `is_healthy`.
    * It can be sync or async def.
    * Must not define any parameters other than `self`.
    * Must return a boolean.
    """
    if not hasattr(cls, private_types.HEALTH_CHECK_METHOD_NAME):
        return None

    health_check_method = getattr(cls, private_types.HEALTH_CHECK_METHOD_NAME)
    if not inspect.isfunction(health_check_method):
        _collect_error(
            f"`{private_types.HEALTH_CHECK_METHOD_NAME}` must be a method.",
            _ErrorKind.TYPE_ERROR,
            location,
        )
        return None

    line = inspect.getsourcelines(health_check_method)[1]
    location = location.model_copy(
        update={"line": line, "method_name": private_types.HEALTH_CHECK_METHOD_NAME}
    )
    is_async = inspect.iscoroutinefunction(health_check_method)
    signature = inspect.signature(health_check_method)
    params = list(signature.parameters.values())
    _validate_method_signature(private_types.HEALTH_CHECK_METHOD_NAME, location, params)
    if len(params) > 1:
        _collect_error(
            f"`{private_types.HEALTH_CHECK_METHOD_NAME}` must have only one argument: `{private_types.SELF_ARG_NAME}`.",
            _ErrorKind.TYPE_ERROR,
            location,
        )
    if signature.return_annotation == inspect.Parameter.empty:
        _collect_error(
            "Return value of health check must be type annotated. Got:\n"
            f"\t{location.method_name}{signature} -> !MISSING!",
            _ErrorKind.IO_TYPE_ERROR,
            location,
        )
        return None
    if signature.return_annotation is not bool:
        _collect_error(
            "Return value of health check must be a boolean. Got:\n"
            f"\t{location.method_name}{signature} -> {signature.return_annotation}",
            _ErrorKind.IO_TYPE_ERROR,
            location,
        )

    return private_types.HealthCheckAPIDescriptor(is_async=is_async)


def validate_and_register_cls(cls: Type[private_types.ABCChainlet]) -> None:
    """Note that validation errors will only be collected, not raised, and Chainlets.
    with issues, are still added to the registry.  Use `raise_validation_errors` to
    assert all Chainlets are valid and before performing operations that depend on
    these constraints."""
    # Skip "abstract" base classes. There's no good way to determine these classes
    # automatically, so we use a list names. It would be good to improve this.
    _skip_class_name = [
        "ChainletBase",
        "ModelBase",
        "EngineBuilderChainlet",
        "EngineBuilderLLMChainlet",
    ]
    if cls.__name__ in _skip_class_name:
        logging.debug(f"Skipping chainlet class validation for `{cls}`.")
        return

    src_path = os.path.abspath(inspect.getfile(cls))
    line = inspect.getsourcelines(cls)[1]
    location = _ErrorLocation(src_path=src_path, line=line, chainlet_name=cls.__name__)

    _validate_display_name(cls, location)
    _validate_config_class_variable(
        cls, location, private_types.REMOTE_CONFIG_NAME, public_types.RemoteConfig
    )
    if is_engine_builder_chainlet(cls):
        _validate_config_class_variable(
            cls,
            location,
            private_types.ENGINE_BUILDER_CONFIG_NAME,
            trt_llm_config.TRTLLMConfiguration,
        )
        _validate_engine_builder_fields(cls.remote_config, location)

    init_validator = _ChainletInitValidator(cls, location)
    chainlet_descriptor = private_types.ChainletAPIDescriptor(
        chainlet_cls=cls,
        dependencies=init_validator.validated_dependencies,
        has_context=init_validator.has_context,
        endpoint=_validate_and_describe_endpoint(cls, location),
        src_path=src_path,
        health_check=_validate_health_check(cls, location),
    )
    logging.debug(
        f"Descriptor for {cls}:\n{pprint.pformat(chainlet_descriptor, indent=4)}\n"
    )
    _global_chainlet_registry.register_chainlet(chainlet_descriptor)


# Dependency-Injection / Registry ######################################################


class _ChainletRegistry:
    # Because dependencies are required to be present when registering a Chainlet,
    # this dict contains natively a topological sorting of the dependency graph.
    _chainlets: collections.OrderedDict[
        Type[private_types.ABCChainlet], private_types.ChainletAPIDescriptor
    ]
    _name_to_cls: MutableMapping[str, Type[private_types.ABCChainlet]]

    def __init__(self) -> None:
        self._chainlets = collections.OrderedDict()
        self._name_to_cls = {}

    def clear(self):
        self._chainlets = collections.OrderedDict()
        self._name_to_cls = {}

    def register_chainlet(
        self, chainlet_descriptor: private_types.ChainletAPIDescriptor
    ):
        for dep in chainlet_descriptor.dependencies.values():
            # To depend on a Chainlet, the class must be defined (module initialized)
            # which entails that is has already been added to the registry.
            # This is an assertion, because unless users meddle with the internal
            # registry, it's not possible to depend on another chainlet before it's
            # also added to the registry.
            assert dep.chainlet_cls in self._chainlets, (
                "Cannot depend on Chainlet. Available Chainlets: "
                f"{list(self._chainlets.keys())}"
            )

        # Because class are globally unique, to prevent re-use / overwriting of names,
        # We must check this in addition.
        if chainlet_descriptor.name in self._name_to_cls:
            conflict = self._name_to_cls[chainlet_descriptor.name]
            existing_source_path = self._chainlets[conflict].src_path
            raise public_types.ChainsUsageError(
                f"A Chainlet with name `{chainlet_descriptor.name}` was already "
                f"defined, Chainlet names must be globally unique.\n"
                f"Pre-existing in: `{existing_source_path}`\n"
                f"New conflict in: `{chainlet_descriptor.src_path}`."
            )

        self._chainlets[chainlet_descriptor.chainlet_cls] = chainlet_descriptor
        self._name_to_cls[chainlet_descriptor.name] = chainlet_descriptor.chainlet_cls

    def unregister_chainlet(self, chainlet_name: str) -> None:
        chainlet_cls = self._name_to_cls.pop(chainlet_name)
        self._chainlets.pop(chainlet_cls)

    @property
    def chainlet_descriptors(self) -> list[private_types.ChainletAPIDescriptor]:
        return list(self._chainlets.values())

    def get_descriptor(
        self, chainlet_cls: Type[private_types.ABCChainlet]
    ) -> private_types.ChainletAPIDescriptor:
        return self._chainlets[chainlet_cls]

    def get_dependencies(
        self, chainlet: private_types.ChainletAPIDescriptor
    ) -> Iterable[private_types.ChainletAPIDescriptor]:
        return [
            self._chainlets[dep.chainlet_cls]
            for dep in self._chainlets[chainlet.chainlet_cls].dependencies.values()
        ]

    def get_chainlet_names(self) -> set[str]:
        return set(self._name_to_cls.keys())


_global_chainlet_registry = _ChainletRegistry()


def get_dependencies(
    chainlet: private_types.ChainletAPIDescriptor,
) -> Iterable[private_types.ChainletAPIDescriptor]:
    return _global_chainlet_registry.get_dependencies(chainlet)


def get_descriptor(
    chainlet_cls: Type[private_types.ABCChainlet],
) -> private_types.ChainletAPIDescriptor:
    return _global_chainlet_registry.get_descriptor(chainlet_cls)


def get_ordered_descriptors() -> list[private_types.ChainletAPIDescriptor]:
    return _global_chainlet_registry.chainlet_descriptors


# Chainlet class runtime utils #########################################################


def _determine_arguments(func: Callable, **kwargs):
    """Merges provided and default arguments to effective invocation arguments."""
    sig = inspect.signature(func)
    bound_args = sig.bind_partial(**kwargs)
    bound_args.apply_defaults()
    return bound_args.arguments


def ensure_args_are_injected(cls, original_init: Callable, kwargs) -> None:
    """Asserts all marker markers are replaced by actual objects."""
    final_args = _determine_arguments(original_init, **kwargs)
    for name, value in final_args.items():
        if name == private_types.CONTEXT_ARG_NAME:
            if not isinstance(value, public_types.DeploymentContext):
                logging.error(
                    f"When initializing {cls.entity_type} `{cls.name}`, for context "
                    f"argument an incompatible value was passed, value: `{value}`."
                )
                raise public_types.ChainsRuntimeError(
                    _instantiation_error_msg(cls.name)
                )
        # The argument is a dependency chainlet.
        elif isinstance(value, _BaseProvisionMarker):
            logging.error(
                f"When initializing {cls.entity_type} `{cls.name}`, for dependency Chainlet"
                f"argument `{name}` an incompatible value was passed, value: `{value}`."
            )
            raise public_types.ChainsRuntimeError(_instantiation_error_msg(cls.name))


# Local Execution ######################################################################

# A variable to track the stack depth relative to `run_local` context manager.
run_local_stack_depth: contextvars.ContextVar[int] = contextvars.ContextVar(
    "run_local_stack_depth"
)

_INIT_LOCAL_NAME = "__init_local__"
_INIT_NAME = "__init__"


def _create_modified_init_for_local(
    chainlet_descriptor: private_types.ChainletAPIDescriptor,
    cls_to_instance: MutableMapping[
        Type[private_types.ABCChainlet], private_types.ABCChainlet
    ],
    secrets: Mapping[str, str],
    data_dir: Optional[pathlib.Path],
    chainlet_to_service: Mapping[str, public_types.DeployedServiceDescriptor],
):
    """Replaces the default argument values with local Chainlet instantiations.

    If this patch is used, Chainlets can be functionally instantiated without
    any init args (because the patched defaults are sufficient).
    """

    def _detect_naive_instantiations(
        stack: list[inspect.FrameInfo], levels_below_run_local: int
    ) -> None:
        # The goal is to find cases where a chainlet is directly instantiated
        # in a place that is not immediately inside the `run_local`-contextmanager.
        # In particular chainlets being instantiated in the `__init__` or `run_remote`
        # methods of other chainlets (instead of being passed as dependencies with
        # `chains.depends()`).
        #
        # We look into the calls stack of any (wrapped) invocation of an
        # ABCChainlet-subclass's `__init__`.
        # We also cut off the "above" call stack, such that `run_local` (and anything
        # above that) is ignored, so it is possible to use `run_local` in nested code.
        #
        # A valid stack looks like this:
        # * `__init_local__` as deepest frame (which would then call
        #   `__init_with_arg_check__` -> `__init__` if validation passes).
        # * If a chainlet has no base classes, this can *only* be called from
        #   `__init_local__` - the part when the chainlet needs to be instantiated and
        #   added to `cls_to_instance`.
        # * If a chainlet has other chainlets as base classes, they may call a chain
        #   of `super().__init()`. Each will add a triple of
        #   (__init__, __init_with_arg_check__, __init_local__) to the stack. While
        #   these 3 init layers belong to the different base classes, the type of the
        #   `self` arg is fixed.
        #
        # To detect invalid stacks we can rephrase this: `__init_local__` can only be
        # called under either of these conditions:
        # * From `__init_local__` when needing to populate `cls_to_instance`.
        # * From a subclass's `__init__` using `super().__init__()`. This means the
        #   type (and instance) of the `self` arg in the calling `__init_local__` and
        #   the invoked `__init__` must are identical. In the forbidden situation that
        #   for example Chainlet `A` tries to create an instance of `B` inside its
        #   `__init__` the `self` args are two different instances.
        substack = stack[:levels_below_run_local]
        parts = ["-------- Chainlet Instantiation Stack --------"]
        # Track the owner classes encountered in the stack to detect invalid scenarios
        transformed_stack = []
        for frame in substack:
            func_name = frame.function
            line_number = frame.lineno
            local_vars = frame.frame.f_locals
            init_owner_class = None
            self_value = None
            # Determine if "self" exists and extract the owner class
            if "self" in local_vars:
                self_value = local_vars["self"]
                if func_name == _INIT_NAME:
                    try:
                        name_parts = frame.frame.f_code.co_qualname.split(".")  # type: ignore[attr-defined]
                    except AttributeError:  # `co_qualname` only in Python 3.11+.
                        name_parts = []
                    if len(name_parts) > 1:
                        init_owner_class = name_parts[-2]
                elif func_name == _INIT_LOCAL_NAME:
                    assert "init_owner_class" in local_vars, (
                        f"`{_INIT_LOCAL_NAME}` must capture `init_owner_class`"
                    )
                    init_owner_class = local_vars["init_owner_class"].__name__

                if init_owner_class:
                    parts.append(
                        f"{func_name}:{line_number} | type(self)=<"
                        f"{self_value.__class__.__name__}> method of <"
                        f"{init_owner_class}>"
                    )
                else:
                    parts.append(
                        f"{func_name}:l{line_number} | type(self)=<"
                        f"{self_value.__class__.__name__}>"
                    )
            else:
                parts.append(f"{func_name}:l{line_number}")

            transformed_stack.append((func_name, self_value, frame))

        if len(parts) > 1:
            logging.debug("\n".join(parts))

        # Analyze the stack after preparing relevant information.
        for i in range(len(transformed_stack) - 1):
            func_name, self_value, _ = transformed_stack[i]
            up_func_name, up_self_value, up_frame = transformed_stack[i + 1]
            if func_name != _INIT_LOCAL_NAME:
                continue  # OK, we only validate `__init_local__` invocations.
            # We are in `__init_local__`. Now check who and how called it.
            if up_func_name == _INIT_LOCAL_NAME:
                # Note: in this case `self` in the current frame is different then
                # self in the parent frame, since a new instance is created.
                continue  # Ok, populating `cls_to_instance`.
            if up_func_name == _INIT_NAME and self_value == up_self_value:
                continue  # OK, call to `super().__init__()`.

            # Everything else is invalid.
            code_context = up_frame.code_context
            assert code_context is not None
            location = (
                f"{up_frame.filename}:{up_frame.lineno} ({up_frame.function})\n"
                f"    {code_context[0].strip()}"
            )
            raise public_types.ChainsRuntimeError(
                _instantiation_error_msg(chainlet_descriptor.name, location)
            )

    __original_init__ = chainlet_descriptor.chainlet_cls.__init__

    @functools.wraps(__original_init__)
    def __init_local__(self: private_types.ABCChainlet, **kwargs) -> None:
        logging.debug(f"Patched `__init__` of `{chainlet_descriptor.name}`.")
        stack_depth = run_local_stack_depth.get(None)
        assert stack_depth is not None, "__init_local__ is only called in context."
        stack = inspect.stack()
        current_stack_depth = len(stack)
        levels_below_run_local = current_stack_depth - stack_depth
        # Capture `init_owner_class` in locals, because we check it in
        # `_detect_naive_instantiations`.
        init_owner_class = chainlet_descriptor.chainlet_cls  # noqa: F841
        _detect_naive_instantiations(stack, levels_below_run_local)

        kwargs_mod = dict(kwargs)
        if (
            chainlet_descriptor.has_context
            and private_types.CONTEXT_ARG_NAME not in kwargs_mod
        ):
            kwargs_mod[private_types.CONTEXT_ARG_NAME] = public_types.DeploymentContext(
                secrets=secrets,
                data_dir=data_dir,
                chainlet_to_service=chainlet_to_service,
            )
        for arg_name, dep in chainlet_descriptor.dependencies.items():
            chainlet_cls = dep.chainlet_cls
            if arg_name in kwargs_mod:
                logging.debug(
                    f"Use given instance for `{arg_name}` of type `{dep.name}`."
                )
                continue
            if chainlet_cls in cls_to_instance:
                logging.debug(
                    f"Use previously created `{arg_name}` of type `{dep.name}`."
                )
                kwargs_mod[arg_name] = cls_to_instance[chainlet_cls]
            else:
                logging.debug(
                    f"Create new instance for `{arg_name}` of type `{dep.name}`. "
                    f"Calling patched __init__."
                )
                assert chainlet_cls.meta_data.init_is_patched
                # Dependency chainlets are instantiated here, using their __init__
                # that is patched for local.
                logging.info(f"Making first {dep.name}.")
                instance = chainlet_cls()  # type: ignore  # Here init args are patched.
                cls_to_instance[chainlet_cls] = instance
                kwargs_mod[arg_name] = instance

        logging.debug(f"Calling original __init__ of {chainlet_descriptor.name}.")
        __original_init__(self, **kwargs_mod)

    return __init_local__


@contextlib.contextmanager
@raise_validation_errors_before
def run_local(
    secrets: Mapping[str, str],
    data_dir: Optional[pathlib.Path],
    chainlet_to_service: Mapping[str, public_types.DeployedServiceDescriptor],
) -> Any:
    """Context to run Chainlets with dependency injection from local instances."""
    type_to_instance: MutableMapping[
        Type[private_types.ABCChainlet], private_types.ABCChainlet
    ] = {}
    original_inits: MutableMapping[Type[private_types.ABCChainlet], Callable] = {}

    # Capture the stack depth when entering the context manager. The stack is used
    # to check that chainlets' `__init__` methods are only called within this context
    # manager, to flag naive instantiations.
    stack_depth = len(inspect.stack())
    for chainlet_descriptor in _global_chainlet_registry.chainlet_descriptors:
        original_inits[chainlet_descriptor.chainlet_cls] = (
            chainlet_descriptor.chainlet_cls.__init__
        )
        init_for_local = _create_modified_init_for_local(
            chainlet_descriptor,
            type_to_instance,
            secrets,
            data_dir,
            chainlet_to_service,
        )
        chainlet_descriptor.chainlet_cls.__init__ = init_for_local  # type: ignore[method-assign]
        chainlet_descriptor.chainlet_cls.meta_data.init_is_patched = True
    # Subtract 2 levels: `run_local` (this) and `__enter__` (from @contextmanager).
    token = run_local_stack_depth.set(stack_depth - 2)
    try:
        yield
    finally:
        # Restore original classes to unpatched state.
        for chainlet_cls, original_init in original_inits.items():
            chainlet_cls.__init__ = original_init  # type: ignore[method-assign]
            chainlet_cls.meta_data.init_is_patched = False

        run_local_stack_depth.reset(token)


########################################################################################


def entrypoint(
    cls_or_chain_name: Optional[Union[Type[ChainletT], str]] = None,
) -> Union[Callable[[Type[ChainletT]], Type[ChainletT]], Type[ChainletT]]:
    """Decorator to tag a Chainlet as an entrypoint.
    Can be used with or without chain name argument.
    """

    def decorator(cls: Type[ChainletT]) -> Type[ChainletT]:
        if not (utils.issubclass_safe(cls, private_types.ABCChainlet)):
            src_path = os.path.abspath(inspect.getfile(cls))
            line = inspect.getsourcelines(cls)[1]
            location = _ErrorLocation(src_path=src_path, line=line)
            _collect_error(
                "Only Chainlet classes can be marked as entrypoint.",
                _ErrorKind.TYPE_ERROR,
                location,
            )
        cls.meta_data.is_entrypoint = True
        if isinstance(cls_or_chain_name, str):
            cls.meta_data.chain_name = cls_or_chain_name
        return cls

    if isinstance(cls_or_chain_name, str):
        return decorator

    assert cls_or_chain_name is not None
    return decorator(cls_or_chain_name)  # Decorator used without arguments


class _ABCImporter(abc.ABC):
    @classmethod
    @abc.abstractmethod
    def _no_entrypoint_error(cls, module_path: pathlib.Path) -> ValueError:
        pass

    @classmethod
    @abc.abstractmethod
    def _multiple_entrypoints_error(
        cls,
        module_path: pathlib.Path,
        entrypoints: set[type[private_types.ABCChainlet]],
    ) -> ValueError:
        pass

    @classmethod
    @abc.abstractmethod
    def _target_cls_type(cls) -> Type[private_types.ABCChainlet]:
        pass

    @classmethod
    def _get_chainlets(
        cls, symbols
    ) -> tuple[
        set[Type[private_types.ABCChainlet]], set[Type[private_types.ABCChainlet]]
    ]:
        chainlets: set[Type[private_types.ABCChainlet]] = {
            sym for sym in symbols if utils.issubclass_safe(sym, cls._target_cls_type())
        }
        entrypoints: set[Type[private_types.ABCChainlet]] = {
            chainlet for chainlet in chainlets if chainlet.meta_data.is_entrypoint
        }
        return chainlets, entrypoints

    @classmethod
    def _load_module(cls, module_path: pathlib.Path) -> tuple[types.ModuleType, Loader]:
        """The context manager ensures that modules imported by the Model/Chain
         are removed upon exit.

        I.e. aiming at making the import idempotent for common usages, although there could
        be additional side effects not accounted for by this implementation."""
        module_name = module_path.stem  # Use the file's name as the module name
        if not os.path.isfile(module_path):
            raise ImportError(
                f"`{module_path}` is not a file. You must point to a python file where "
                f"the entrypoint is defined."
            )

        import_error_msg = f"Could not import `{module_path}`. Check path."
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if not spec or not spec.loader:
            raise ImportError(import_error_msg)

        module = importlib.util.module_from_spec(spec)
        module.__file__ = str(module_path)
        # Since the framework depends on tracking the source files via `inspect` and this
        # depends on the modules bein properly registered in `sys.modules`, we have to
        # manually do this here (because importlib does not do it automatically). This
        # registration has to stay at least until the push command has finished.
        if module_name in sys.modules:
            raise ImportError(
                f"{import_error_msg} There is already a module in `sys.modules` "
                f"with name `{module_name}`. Overwriting that value is unsafe. "
                "Try renaming your source file."
            )

        sys.modules[module_name] = module
        # Add path for making absolute imports relative to the source_module's dir.
        sys.path.insert(0, str(module_path.parent))

        return module, spec.loader

    @classmethod
    def _cleanup_module_imports(
        cls,
        modules_before: set[str],
        modules_after: set[str],
        module_path: pathlib.Path,
    ):
        modules_diff = modules_after - modules_before
        # Apparently torch import leaves some side effects that cannot be reverted
        # by deleting the modules and would lead to a crash when another import
        # is attempted. Since torch is a common lib, we make this explicit special
        # case and just leave those modules.
        # TODO: this seems still brittle and other modules might cause similar problems.
        #  it would be good to find a more principled solution.
        modules_to_delete = {
            s for s in modules_diff if not (s.startswith("torch.") or s == "torch")
        }
        if torch_modules := modules_diff - modules_to_delete:
            logging.debug(
                f"Keeping torch modules after import context: {torch_modules}"
            )

        logging.debug(
            f"Deleting modules when exiting import context: {modules_to_delete}"
        )
        for mod in modules_to_delete:
            del sys.modules[mod]
        try:
            sys.path.remove(str(module_path.parent))
        except ValueError:  # In case the value was already removed for whatever reason.
            pass

    @classmethod
    @contextlib.contextmanager
    def import_target(
        cls, module_path: pathlib.Path, target_name: Optional[str] = None
    ) -> Iterator[Type[private_types.ABCChainlet]]:
        resolved_module_path = pathlib.Path(module_path).resolve()
        modules_before = set(sys.modules.keys())
        module, loader = cls._load_module(module_path)
        modules_after = set()

        chainlets_before = _global_chainlet_registry.get_chainlet_names()
        chainlets_after = set()
        try:
            try:
                loader.exec_module(module)
                raise_validation_errors()
            finally:
                modules_after = set(sys.modules.keys())
                chainlets_after = _global_chainlet_registry.get_chainlet_names()

            if target_name:
                target_cls = getattr(module, target_name, None)
                if not target_cls:
                    raise AttributeError(
                        f"Target class `{target_name}` not found "
                        f"in `{resolved_module_path}`."
                    )
                if not utils.issubclass_safe(target_cls, cls._target_cls_type()):
                    raise TypeError(
                        f"Target `{target_cls}` is not a {cls._target_cls_type()}."
                    )
            else:
                module_vars = (getattr(module, name) for name in dir(module))
                chainlets, entrypoints = cls._get_chainlets(module_vars)
                if len(chainlets) == 1:
                    entrypoints = chainlets

                if len(entrypoints) == 0:
                    raise cls._no_entrypoint_error(module_path)
                elif len(entrypoints) > 1:
                    raise cls._multiple_entrypoints_error(module_path, entrypoints)
                target_cls = utils.expect_one(entrypoints)

            yield target_cls
        finally:
            cls._cleanup_module_imports(
                modules_before, modules_after, resolved_module_path
            )
            for chainlet_name in chainlets_after - chainlets_before:
                _global_chainlet_registry.unregister_chainlet(chainlet_name)


class ChainletImporter(_ABCImporter):
    @classmethod
    def _no_entrypoint_error(cls, module_path: pathlib.Path) -> ValueError:
        return ValueError(
            "No `target_name` was specified and no Chainlet in "
            f"`{module_path}` was tagged with `@chains.mark_entrypoint`. Tag "
            "one Chainlet or provide the Chainlet class name."
        )

    @classmethod
    def _multiple_entrypoints_error(
        cls,
        module_path: pathlib.Path,
        entrypoints: set[type[private_types.ABCChainlet]],
    ) -> ValueError:
        return ValueError(
            "`target_name` was not specified and multiple Chainlets in "
            f"`{module_path}` were tagged with `@chains.mark_entrypoint`. Tag "
            "one Chainlet or provide the Chainlet class name. Found Chainlets: "
            f"\n{list(cls.name for cls in entrypoints)}"
        )

    @classmethod
    def _target_cls_type(cls) -> Type[private_types.ABCChainlet]:
        return ChainletBase


class ModelImporter(_ABCImporter):
    @classmethod
    def _no_entrypoint_error(cls, module_path: pathlib.Path) -> ValueError:
        return ValueError(
            f"No Model class in `{module_path}` inherits from {cls._target_cls_type()}."
        )

    @classmethod
    def _multiple_entrypoints_error(
        cls,
        module_path: pathlib.Path,
        entrypoints: set[type[private_types.ABCChainlet]],
    ) -> ValueError:
        return ValueError(
            f"Multiple Model classes in `{module_path}` inherit from {cls._target_cls_type()}, "
            "but only one allowed. Found classes: "
            f"\n{list(cls.name for cls in entrypoints)}"
        )

    @classmethod
    def _target_cls_type(cls) -> Type[private_types.ABCChainlet]:
        return ModelBase


class ChainletBase(private_types.ABCChainlet, metaclass=abc.ABCMeta):
    """Base class for all chainlets.

    Inheriting from this class adds validations to make sure subclasses adhere to the
    chainlet pattern and facilitates remote chainlet deployment.

    Refer to `the docs <https://docs.baseten.co/chains/getting-started>`_ and this
    `example chainlet <https://github.com/basetenlabs/truss/blob/main/truss-chains/truss_chains/reference_code/reference_chainlet.py>`_
    for more guidance on how to create subclasses.
    """

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        cls._framework_config = private_types.FrameworkConfig(
            entity_type=private_types.EntityType.CHAINLET,
            supports_dependencies=True,
            endpoint_method_name=private_types.RUN_REMOTE_METHOD_NAME,
        )
        # Each sub-class has own, isolated metadata, e.g. we don't want
        # `mark_entrypoint` to propagate to subclasses.
        cls.meta_data = private_types.ChainletMetadata()
        validate_and_register_cls(cls)  # Errors are collected, not raised!
        # For default init (from `object`) we don't need to check anything.
        if cls.has_custom_init():
            original_init = cls.__init__

            @functools.wraps(original_init)
            def __init_with_arg_check__(self, *args, **kwargs):
                if args:
                    raise public_types.ChainsRuntimeError("Only kwargs are allowed.")
                ensure_args_are_injected(cls, original_init, kwargs)
                original_init(self, *args, **kwargs)

            cls.__init__ = __init_with_arg_check__  # type: ignore[method-assign]


class ModelBase(private_types.ABCChainlet, metaclass=abc.ABCMeta):
    """Base class for all standalone models.

    Inheriting from this class adds validations to make sure subclasses adhere to the
    truss model pattern.
    """

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        cls._framework_config = private_types.FrameworkConfig(
            entity_type=private_types.EntityType.MODEL,
            supports_dependencies=False,
            endpoint_method_name=private_types.MODEL_ENDPOINT_METHOD_NAME,
        )
        cls.meta_data = private_types.ChainletMetadata(is_entrypoint=True)
        validate_and_register_cls(cls)


class EngineBuilderChainlet(private_types.ABCChainlet, metaclass=abc.ABCMeta):
    """For engine builders, model.py is generated during deployment, so there is only
    dummy `run_remote` and we do not generate `model.py` wrapping the chainlet.
    We do not support customization, because that should be done caller-side for chains.
    """

    engine_builder_config: ClassVar[trt_llm_config.TRTLLMConfiguration]

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        cls._framework_config = private_types.FrameworkConfig(
            entity_type=private_types.EntityType.ENGINE_BUILDER_MODEL,
            supports_dependencies=False,
            endpoint_method_name=private_types.RUN_REMOTE_METHOD_NAME,
        )
        validate_and_register_cls(cls)


class EngineBuilderLLMChainlet(EngineBuilderChainlet, metaclass=abc.ABCMeta):
    @final
    async def run_remote(
        self, llm_input: public_types.EngineBuilderLLMInput
    ) -> AsyncIterator[str]:
        raise NotImplementedError("Only deployed models generate output.")
        yield


def is_engine_builder_chainlet(cls: Type[private_types.ABCChainlet]):
    return issubclass(cls, EngineBuilderChainlet)
