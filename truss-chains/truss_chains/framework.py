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
import sys
import types
import warnings
from typing import (
    Any,
    Callable,
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    Optional,
    Protocol,
    Type,
    TypeVar,
    get_args,
    get_origin,
)

import pydantic
from typing_extensions import ParamSpec

from truss_chains import definitions, utils

_SIMPLE_TYPES = {int, float, complex, bool, str, bytes, None}
_SIMPLE_CONTAINERS = {list, dict}

_DOCS_URL_CHAINING = (
    "https://docs.baseten.co/chains/concepts#depends-call-other-chainlets"
)
_DOCS_URL_LOCAL = "https://docs.baseten.co/chains/guide#local-development"

_ENTRYPOINT_ATTR_NAME = "_chains_entrypoint"

ChainletT = TypeVar("ChainletT", bound=definitions.ABCChainlet)
_P = ParamSpec("_P")
_R = TypeVar("_R")

# Error Collector ######################################################################


class _ErrorKind(str, enum.Enum):
    TYPE_ERROR = enum.auto()
    IO_TYPE_ERROR = enum.auto()
    MISSING_API_ERROR = enum.auto()


class _ErrorLocation(definitions.SafeModel):
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


class _ValidationError(definitions.SafeModel):
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
        errors_count = (
            "an error"
            if _global_error_collector.num_errors == 1
            else f"{_global_error_collector.num_errors} errors"
        )
        _global_error_collector.clear()  # Clear errors so `atexit` won't display them
        raise definitions.ChainsUsageError(
            f"The Chainlet definitions contain {errors_count}:\n{error_msg}"
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
        raise definitions.ChainsRuntimeError(
            "It seems `chains.depends_context()` was used, but not as an argument "
            "to the `__init__` method of a Chainlet - This is not supported."
            f"See {_DOCS_URL_CHAINING}.\n"
            "Example of correct `__init__` with context:\n"
            f"{_example_chainlet_code()}"
        )


class ChainletDependencyMarker(_BaseProvisionMarker):
    chainlet_cls: Type[definitions.ABCChainlet]
    retries: int

    def __init__(
        self,
        chainlet_cls: Type[definitions.ABCChainlet],
        options: definitions.RPCOptions,
    ) -> None:
        self.chainlet_cls = chainlet_cls
        self.options = options

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.chainlet_cls.name})"

    def __getattr__(self, item: str) -> Any:
        logging.error(f"Attempting to access attribute `{item}` on `{self}`.")
        raise definitions.ChainsRuntimeError(
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
        from truss_chains import example_chainlet
    # If `example_chainlet` fails validation and `_example_chainlet_code` is
    # called as a result of that, we have a circular import ("partially initialized
    # module 'truss_chains.example_chainlet' ...").
    except AttributeError:
        logging.error("example_chainlet` is broken.", exc_info=True, stack_info=True)
        return "<EXAMPLE CODE MISSING/BROKEN>"

    example_name = example_chainlet.HelloWorld.name
    source = pathlib.Path(example_chainlet.__file__).read_text()
    tree = ast.parse(source)
    class_code = ""
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == example_name:
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
    if annotation in _SIMPLE_TYPES:
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
            if arg not in _SIMPLE_TYPES:
                _collect_error(error_msg, _ErrorKind.IO_TYPE_ERROR, location)
                return
        return
    if utils.issubclass_safe(annotation, pydantic.BaseModel):
        return

    _collect_error(error_msg, _ErrorKind.IO_TYPE_ERROR, location)


def _validate_endpoint_params(
    params: list[inspect.Parameter], location: _ErrorLocation
) -> list[definitions.InputArg]:
    if len(params) == 0:
        _collect_error(
            f"`Endpoint must be a method, i.e. with `{definitions.SELF_ARG_NAME}` as "
            "first argument. Got function with no arguments.",
            _ErrorKind.TYPE_ERROR,
            location,
        )
        return []
    if params[0].name != definitions.SELF_ARG_NAME:
        _collect_error(
            f"`Endpoint must be a method, i.e. with `{definitions.SELF_ARG_NAME}` as "
            f"first argument. Got `{params[0].name}` as first argument.",
            _ErrorKind.TYPE_ERROR,
            location,
        )
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
            type_descriptor = definitions.TypeDescriptor(raw=param.annotation)
            is_optional = param.default != inspect.Parameter.empty
            input_args.append(
                definitions.InputArg(
                    name=param.name, type=type_descriptor, is_optional=is_optional
                )
            )
    return input_args


def _validate_endpoint_output_types(
    annotation: Any, signature, location: _ErrorLocation
) -> list[definitions.TypeDescriptor]:
    if annotation == inspect.Parameter.empty:
        _collect_error(
            "Return values of endpoints must be type annotated. Got:\n"
            f"\t{location.method_name}{signature} -> !MISSING!",
            _ErrorKind.IO_TYPE_ERROR,
            location,
        )
        return []
    if get_origin(annotation) is tuple:
        output_types = []
        for i, arg in enumerate(get_args(annotation)):
            _validate_io_type(arg, f"return_type[{i}]", location)
            output_types.append(definitions.TypeDescriptor(raw=arg))
    else:
        _validate_io_type(annotation, "return_type", location)
        output_types = [definitions.TypeDescriptor(raw=annotation)]
    return output_types


def _validate_and_describe_endpoint(
    cls: Type[definitions.ABCChainlet], location: _ErrorLocation
) -> definitions.EndpointAPIDescriptor:
    """The "endpoint method" of a Chainlet must have the following signature:

    ```
    [async] def run_remote(
        self, [param_0: anno_0, param_1: anno_1 = default_1, ...]) -> ret_anno:
    ```

    * The name must be `run`.
    * It can be sync or async or def.
    * The number and names of parameters are arbitrary, both positional and named
      parameters are ok.
    * All parameters and the return value must have type annotations. See
      `_validate_io_type` for valid types.
    * Generators are allowed, too (but not yet supported).
    """
    if not hasattr(cls, definitions.ENDPOINT_METHOD_NAME):
        _collect_error(
            f"Chainlets must have a `{definitions.ENDPOINT_METHOD_NAME}` method.",
            _ErrorKind.MISSING_API_ERROR,
            location,
        )
        # Return a "neutral dummy" if validation fails, this allows to safely
        # continue checking for more errors.
        return definitions.EndpointAPIDescriptor(
            input_args=[], output_types=[], is_async=False, is_generator=False
        )

    # This is the unbound method.
    endpoint_method = getattr(cls, definitions.ENDPOINT_METHOD_NAME)

    line = inspect.getsourcelines(endpoint_method)[1]
    location = location.model_copy(
        update={"line": line, "method_name": definitions.ENDPOINT_METHOD_NAME}
    )

    if not inspect.isfunction(endpoint_method):
        _collect_error("`Endpoints must be a method.", _ErrorKind.TYPE_ERROR, location)
        # If it's not a function, it might be a class var and subsequent inspections
        # fail.
        # Return a "neutral dummy" if validation fails, this allows to safely
        # continue checking for more errors.
        return definitions.EndpointAPIDescriptor(
            input_args=[], output_types=[], is_async=False, is_generator=False
        )
    signature = inspect.signature(endpoint_method)
    input_args = _validate_endpoint_params(
        list(signature.parameters.values()), location
    )

    output_types = _validate_endpoint_output_types(
        signature.return_annotation, signature, location
    )

    if inspect.isasyncgenfunction(endpoint_method):
        is_async = True
        is_generator = True
    elif inspect.iscoroutinefunction(endpoint_method):
        is_async = True
        is_generator = False
    else:
        is_async = False
        is_generator = inspect.isgeneratorfunction(endpoint_method)

    if not is_async:
        warnings.warn(
            "`run_remote` must be an async (coroutine) function in future releases. "
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

    return definitions.EndpointAPIDescriptor(
        input_args=input_args,
        output_types=output_types,
        is_async=is_async,
        is_generator=is_generator,
    )


def _get_generic_class_type(var):
    """Extracts `SomeGeneric` from `SomeGeneric` or `SomeGeneric[T]` uniformly."""
    origin = get_origin(var)
    return origin if origin is not None else var


def _validate_dependency_arg(
    param, location: _ErrorLocation
) -> Optional[ChainletDependencyMarker]:
    # Returns `None` if unvalidated.
    # TODO: handle subclasses, unions, optionals, check default value etc.
    if param.name == definitions.CONTEXT_ARG_NAME:
        _collect_error(
            f"The init argument name `{definitions.CONTEXT_ARG_NAME}` is reserved for "
            "the optional context argument, which must be trailing if used. Example "
            "of correct `__init__` with context:\n"
            f"{_example_chainlet_code()}",
            _ErrorKind.TYPE_ERROR,
            location,
        )

    if not isinstance(param.default, ChainletDependencyMarker):
        _collect_error(
            f"Any arguments of a Chainlet's __init__ (besides `context`) must have "
            "dependency Chainlets with default values from `chains.depends`-directive. "
            f"Got `{param}`.\n"
            f"Example of correct `__init__` with dependencies:\n"
            f"{_example_chainlet_code()}",
            _ErrorKind.TYPE_ERROR,
            location,
        )
        return None

    chainlet_cls = param.default.chainlet_cls
    if not utils.issubclass_safe(chainlet_cls, definitions.ABCChainlet):
        _collect_error(
            f"`chains.depends` must be used with a Chainlet class as argument, got "
            f"{chainlet_cls} instead.",
            _ErrorKind.TYPE_ERROR,
            location,
        )
        return None
    # Check type annotation.
    # Also lenient with type annotation: since the RHS / default is asserted to be a
    # chainlet class, proper type inference is possible even without annotation.
    # TODO: `Protocol` is not a proper class and this might be version dependent.
    # Find a better way to inspect this.
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
            location,
        )
    return param.default  # The Marker.


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
    has_context: bool
    validated_dependencies: Mapping[str, definitions.DependencyDescriptor]

    def __init__(
        self, cls: Type[definitions.ABCChainlet], location: _ErrorLocation
    ) -> None:
        if not cls.has_custom_init():
            self.has_context = False
            self.validated_dependencies = {}
            return
        # Each validation pops of "processed" arguments from the list.
        line = inspect.getsourcelines(cls.__init__)[1]
        self._location = location.model_copy(
            update={"line": line, "method_name": "__init__"}
        )
        params = list(inspect.signature(cls.__init__).parameters.values())
        params = self._validate_self_arg(list(params))
        params, self.has_context = self._validate_context_arg(params)
        self.validated_dependencies = self._validate_dependencies(params)

    def _validate_self_arg(
        self, params: list[inspect.Parameter]
    ) -> list[inspect.Parameter]:
        if len(params) == 0:
            _collect_error(
                "Methods must have first argument `self`, got no arguments.",
                _ErrorKind.TYPE_ERROR,
                self._location,
            )
            return params
        param = params.pop(0)
        if param.name != definitions.SELF_ARG_NAME:
            _collect_error(
                f"Methods must have first argument `self`, got `{param.name}`.",
                _ErrorKind.TYPE_ERROR,
                self._location,
            )
        return params

    def _validate_context_arg(
        self, params: list[inspect.Parameter]
    ) -> tuple[list[inspect.Parameter], bool]:
        def make_context_error_msg():
            return (
                f"If `{definitions.ABCChainlet}` uses context for initialization, it "
                f"must have `{definitions.CONTEXT_ARG_NAME}` argument of type "
                f"`{definitions.DeploymentContext}` as the last argument.\n"
                f"Got arguments: `{params}`.\n"
                "Example of correct `__init__` with context:\n"
                f"{_example_chainlet_code()}"
            )

        if not params:
            return params, False

        has_context = params[-1].name == definitions.CONTEXT_ARG_NAME
        has_context_marker = isinstance(params[-1].default, ContextDependencyMarker)
        if has_context ^ has_context_marker:
            _collect_error(
                make_context_error_msg(), _ErrorKind.TYPE_ERROR, self._location
            )

        if not has_context:
            return params, has_context

        param = params.pop(-1)
        param_type = _get_generic_class_type(param.annotation)
        # We are lenient and allow omitting the type annotation for context.
        if (
            (param_type is not None)
            and (param_type != inspect.Parameter.empty)
            and (not utils.issubclass_safe(param_type, definitions.DeploymentContext))
        ):
            _collect_error(
                make_context_error_msg(), _ErrorKind.TYPE_ERROR, self._location
            )
        if not isinstance(param.default, ContextDependencyMarker):
            _collect_error(
                f"Incorrect default value `{param.default}` for `context` argument. "
                "Example of correct `__init__` with dependencies:\n"
                f"{_example_chainlet_code()}",
                _ErrorKind.TYPE_ERROR,
                self._location,
            )

        return params, has_context

    def _validate_dependencies(
        self, params
    ) -> Mapping[str, definitions.DependencyDescriptor]:
        used = set()
        dependencies = {}
        for param in params:
            marker = _validate_dependency_arg(param, self._location)
            if marker is None:
                continue
            if marker.chainlet_cls in used:
                _collect_error(
                    f"The same Chainlet class cannot be used multiple times for "
                    f"different arguments. Got previously used "
                    f"`{marker.chainlet_cls}` for `{param.name}`.",
                    _ErrorKind.TYPE_ERROR,
                    self._location,
                )

            dependencies[param.name] = definitions.DependencyDescriptor(
                chainlet_cls=marker.chainlet_cls, options=marker.options
            )
            used.add(marker.chainlet_cls)
        return dependencies


def _validate_chainlet_cls(
    cls: Type[definitions.ABCChainlet], location: _ErrorLocation
) -> None:
    if not hasattr(cls, definitions.REMOTE_CONFIG_NAME):
        _collect_error(
            f"Chainlets must have a `{definitions.REMOTE_CONFIG_NAME}` class variable "
            f"`{definitions.REMOTE_CONFIG_NAME} = {definitions.RemoteConfig.__name__}"
            f"(...)`. Missing for `{cls}`.",
            _ErrorKind.MISSING_API_ERROR,
            location,
        )
        return

    if not isinstance(
        remote_config := getattr(cls, definitions.REMOTE_CONFIG_NAME),
        definitions.RemoteConfig,
    ):
        _collect_error(
            f"Chainlets must have a `{definitions.REMOTE_CONFIG_NAME}` class variable "
            f"of type `{definitions.RemoteConfig}`. Got `{type(remote_config)}` "
            f"for `{cls}`.",
            _ErrorKind.TYPE_ERROR,
            location,
        )
        return


def validate_and_register_class(cls: Type[definitions.ABCChainlet]) -> None:
    """Note that validation errors will only be collected, not raised, and Chainlets.
    with issues, are still added to the registry.  Use `raise_validation_errors` to
    assert all Chainlets are valid and before performing operations that depend on
    these constraints."""
    src_path = os.path.abspath(inspect.getfile(cls))
    line = inspect.getsourcelines(cls)[1]
    location = _ErrorLocation(src_path=src_path, line=line, chainlet_name=cls.__name__)

    _validate_chainlet_cls(cls, location)
    init_validator = _ChainletInitValidator(cls, location)
    chainlet_descriptor = definitions.ChainletAPIDescriptor(
        chainlet_cls=cls,
        dependencies=init_validator.validated_dependencies,
        has_context=init_validator.has_context,
        endpoint=_validate_and_describe_endpoint(cls, location),
        src_path=src_path,
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
        Type[definitions.ABCChainlet], definitions.ChainletAPIDescriptor
    ]
    _name_to_cls: MutableMapping[str, Type[definitions.ABCChainlet]]

    def __init__(self) -> None:
        self._chainlets = collections.OrderedDict()
        self._name_to_cls = {}

    def clear(self):
        self._chainlets = collections.OrderedDict()
        self._name_to_cls = {}

    def register_chainlet(self, chainlet_descriptor: definitions.ChainletAPIDescriptor):
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
            raise definitions.ChainsUsageError(
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
    def chainlet_descriptors(self) -> list[definitions.ChainletAPIDescriptor]:
        return list(self._chainlets.values())

    def get_descriptor(
        self, chainlet_cls: Type[definitions.ABCChainlet]
    ) -> definitions.ChainletAPIDescriptor:
        return self._chainlets[chainlet_cls]

    def get_dependencies(
        self, chainlet: definitions.ChainletAPIDescriptor
    ) -> Iterable[definitions.ChainletAPIDescriptor]:
        return [
            self._chainlets[dep.chainlet_cls]
            for dep in self._chainlets[chainlet.chainlet_cls].dependencies.values()
        ]

    def get_chainlet_names(self) -> set[str]:
        return set(self._name_to_cls.keys())


_global_chainlet_registry = _ChainletRegistry()


def get_dependencies(
    chainlet: definitions.ChainletAPIDescriptor,
) -> Iterable[definitions.ChainletAPIDescriptor]:
    return _global_chainlet_registry.get_dependencies(chainlet)


def get_descriptor(
    chainlet_cls: Type[definitions.ABCChainlet],
) -> definitions.ChainletAPIDescriptor:
    return _global_chainlet_registry.get_descriptor(chainlet_cls)


def get_ordered_descriptors() -> list[definitions.ChainletAPIDescriptor]:
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
        if name == definitions.CONTEXT_ARG_NAME:
            if not isinstance(value, definitions.DeploymentContext):
                logging.error(
                    f"When initializing Chainlet `{cls.name}`, for context "
                    f"argument an incompatible value was passed, value: `{value}`."
                )
                raise definitions.ChainsRuntimeError(_instantiation_error_msg(cls.name))
        # The argument is a dependency chainlet.
        elif isinstance(value, _BaseProvisionMarker):
            logging.error(
                f"When initializing Chainlet `{cls.name}`, for dependency Chainlet"
                f"argument `{name}` an incompatible value was passed, value: `{value}`."
            )
            raise definitions.ChainsRuntimeError(_instantiation_error_msg(cls.name))


# Local Execution ######################################################################

# A variable to track the stack depth relative to `run_local` context manager.
run_local_stack_depth: contextvars.ContextVar[int] = contextvars.ContextVar(
    "run_local_stack_depth"
)

_INIT_LOCAL_NAME = "__init_local__"
_INIT_NAME = "__init__"


def _create_modified_init_for_local(
    chainlet_descriptor: definitions.ChainletAPIDescriptor,
    cls_to_instance: MutableMapping[
        Type[definitions.ABCChainlet], definitions.ABCChainlet
    ],
    secrets: Mapping[str, str],
    data_dir: Optional[pathlib.Path],
    chainlet_to_service: Mapping[str, definitions.ServiceDescriptor],
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
                    assert (
                        "init_owner_class" in local_vars
                    ), f"`{_INIT_LOCAL_NAME}` must capture `init_owner_class`"
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
            location = (
                f"{up_frame.filename}:{up_frame.lineno} ({up_frame.function})\n"
                f"    {up_frame.code_context[0].strip()}"  # type: ignore[index]
            )
            raise definitions.ChainsRuntimeError(
                _instantiation_error_msg(chainlet_descriptor.name, location)
            )

    __original_init__ = chainlet_descriptor.chainlet_cls.__init__

    @functools.wraps(__original_init__)
    def __init_local__(self: definitions.ABCChainlet, **kwargs) -> None:
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
            and definitions.CONTEXT_ARG_NAME not in kwargs_mod
        ):
            kwargs_mod[definitions.CONTEXT_ARG_NAME] = definitions.DeploymentContext(
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
                assert chainlet_cls._init_is_patched
                # Dependency chainlets are instantiated here, using their __init__
                # that is patched for local.
                logging.warning(f"Making first {dep.name}.")
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
    chainlet_to_service: Mapping[str, definitions.ServiceDescriptor],
) -> Any:
    """Context to run Chainlets with dependency injection from local instances."""
    # TODO: support retries in local mode.
    type_to_instance: MutableMapping[
        Type[definitions.ABCChainlet], definitions.ABCChainlet
    ] = {}
    original_inits: MutableMapping[Type[definitions.ABCChainlet], Callable] = {}

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
        chainlet_descriptor.chainlet_cls._init_is_patched = True
    # Subtract 2 levels: `run_local` (this) and `__enter__` (from @contextmanager).
    token = run_local_stack_depth.set(stack_depth - 2)
    try:
        yield
    finally:
        # Restore original classes to unpatched state.
        for chainlet_cls, original_init in original_inits.items():
            chainlet_cls.__init__ = original_init  # type: ignore[method-assign]
            chainlet_cls._init_is_patched = False

        run_local_stack_depth.reset(token)


########################################################################################


def entrypoint(cls: Type[ChainletT]) -> Type[ChainletT]:
    """Decorator to tag a Chainlet as an entrypoint."""
    if not (utils.issubclass_safe(cls, definitions.ABCChainlet)):
        src_path = os.path.abspath(inspect.getfile(cls))
        line = inspect.getsourcelines(cls)[1]
        location = _ErrorLocation(src_path=src_path, line=line)
        _collect_error(
            "Only Chainlet classes can be marked as entrypoint.",
            _ErrorKind.TYPE_ERROR,
            location,
        )
    setattr(cls, _ENTRYPOINT_ATTR_NAME, True)
    return cls


def _get_entrypoint_chainlets(symbols) -> set[Type[definitions.ABCChainlet]]:
    return {
        sym
        for sym in symbols
        if utils.issubclass_safe(sym, definitions.ABCChainlet)
        and getattr(sym, _ENTRYPOINT_ATTR_NAME, False)
    }


@contextlib.contextmanager
def import_target(
    module_path: pathlib.Path, target_name: Optional[str]
) -> Iterator[Type[definitions.ABCChainlet]]:
    """The context manager ensures that modules imported by the chain and
    Chainlets registered in ``_global_chainlet_registry`` are removed upon exit.

    I.e. aiming at making the import idempotent for common usages, although there could
    be additional side effects not accounted for by this implementation."""
    module_path = pathlib.Path(module_path).resolve()
    module_name = module_path.stem  # Use the file's name as the module name
    if not os.path.isfile(module_path):
        raise ImportError(
            f"`{module_path}` is not a file. You must point to a python file where "
            "the entrypoint Chainlet is defined."
        )

    import_error_msg = f"Could not import `{module_path}`. Check path."
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if not spec:
        raise ImportError(import_error_msg)
    if not spec.loader:
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
    modules_before = set(sys.modules.keys())
    sys.modules[module_name] = module
    # Add path for making absolute imports relative to the source_module's dir.
    sys.path.insert(0, str(module_path.parent))
    chainlets_before = _global_chainlet_registry.get_chainlet_names()
    chainlets_after = set()
    modules_after = set()
    try:
        try:
            spec.loader.exec_module(module)
            raise_validation_errors()
        finally:
            modules_after = set(sys.modules.keys())
            chainlets_after = _global_chainlet_registry.get_chainlet_names()

        if target_name:
            target_cls = getattr(module, target_name, None)
            if not target_cls:
                raise AttributeError(
                    f"Target Chainlet class `{target_name}` not found "
                    f"in `{module_path}`."
                )
            if not utils.issubclass_safe(target_cls, definitions.ABCChainlet):
                raise TypeError(
                    f"Target `{target_cls}` is not a {definitions.ABCChainlet}."
                )
        else:
            module_vars = (getattr(module, name) for name in dir(module))
            entrypoints = _get_entrypoint_chainlets(module_vars)
            if len(entrypoints) == 0:
                raise ValueError(
                    "No `target_name` was specified and no Chainlet in "
                    "`{module_path}` was tagged with `@chains.mark_entrypoint`. Tag "
                    "one Chainlet or provide the Chainlet class name."
                )
            elif len(entrypoints) > 1:
                raise ValueError(
                    "`target_name` was not specified and multiple Chainlets in "
                    f"`{module_path}` were tagged with `@chains.mark_entrypoint`. Tag "
                    "one Chainlet or provide the Chainlet class name. Found Chainlets: "
                    f"\n{entrypoints}"
                )
            target_cls = utils.expect_one(entrypoints)
            if not utils.issubclass_safe(target_cls, definitions.ABCChainlet):
                raise TypeError(
                    f"Target `{target_cls}` is not a {definitions.ABCChainlet}."
                )

        yield target_cls
    finally:
        for chainlet_name in chainlets_after - chainlets_before:
            _global_chainlet_registry.unregister_chainlet(chainlet_name)

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
