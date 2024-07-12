import ast
import collections
import contextlib
import contextvars
import functools
import importlib.util
import inspect
import logging
import os
import pathlib
import sys
import types
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

from truss_chains import definitions, utils

_SIMPLE_TYPES = {int, float, complex, bool, str, bytes, None}
_SIMPLE_CONTAINERS = {list, dict}

_DOCS_URL_CHAINING = "https://docs.baseten.co/chains/chaining-chainlets"
_DOCS_URL_LOCAL = "https://docs.baseten.co/chains/gettin-started"

_ENTRYPOINT_ATTR_NAME = "_chains_entrypoint"

ChainletT = TypeVar("ChainletT", bound=definitions.ABCChainlet)


class _BaseProvisionMarker:
    """A marker for object to be dependency injected by the framework."""


class ContextDependencyMarker(_BaseProvisionMarker):
    def __str__(self) -> str:
        return f"{self.__class__.__name__}"

    def __getattr__(self, item: str) -> Any:
        logging.error(f"Attempting to access attribute `{item}` on `{self}`.")
        raise definitions.ChainsRuntimeError(
            "It seems `chains.depends_context()` was used, but not as an argument "
            "to the `__init__` method of a chainlet - This is not supported."
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
        return f"{self.__class__.__name__}({self.chainlet_cls.__name__})"

    def __getattr__(self, item: str) -> Any:
        logging.error(f"Attempting to access attribute `{item}` on `{self}`.")
        raise definitions.ChainsRuntimeError(
            f"It seems `chains.depends({self.chainlet_cls.__name__})` was used, but "
            "not as an argument to the `__init__` method of a chainlet - This is not "
            "supported. Dependency chainlets must be passed as init arguments.\n"
            f"See {_DOCS_URL_CHAINING}.\n"
            "Example of correct `__init__` with dependencies:\n"
            f"{_example_chainlet_code()}"
        )


# Checking of Chainlet class definition ###############################################


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

    example_name = example_chainlet.HelloWorld.__name__
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


def _instantiation_error_msg(cls_name: str):
    return (
        f"Error when instantiating chainlet `{cls_name}`. "
        "Chainlets cannot be naively instantiated. Possible fixes:\n"
        "1. To use chainlets as dependencies in other chainlets 'chaining'), "
        f"add them as init argument. See {_DOCS_URL_CHAINING}.\n"
        f"2. For local / debug execution, use the `{run_local.__name__}`-"
        f"context. See {_DOCS_URL_LOCAL}.\n"
        "3. Deploy the chain and call the remote endpoint.\n"
        "Example of correct `__init__` with dependencies:\n"
        f"{_example_chainlet_code()}"
    )


def _validate_io_type(param: inspect.Parameter) -> None:
    """
    For Chainlet I/O (both data or parameters), we allow simple types
    (int, str, float...) and `list` or `dict` containers of these.
    Any deeper nested and structured data must be typed as a pydantic model.
    """
    containers_str = [c.__name__ for c in _SIMPLE_CONTAINERS]
    types_str = [c.__name__ if c is not None else "None" for c in _SIMPLE_TYPES]
    error_msg = (
        f"Unsupported I/O type `{param}`. Supported are:\n"
        f"\t* simple types: {types_str}\n"
        f"\t* containers of these simple types, with annotated items: {containers_str}"
        ", e.g. `dict[str, int]` (use built-in types, not `typing.Dict`).\n"
        "\t* For complicated / nested data structures: `pydantic` models."
    )
    anno = param.annotation
    if isinstance(anno, str):
        raise definitions.ChainsUsageError(
            f"A string-valued type annotation was found: `{param}`. Use only actual "
            "types and avoid `from __future__ import annotations` (upgrade python)."
        )
    if anno in _SIMPLE_TYPES:
        return
    if isinstance(anno, types.GenericAlias):
        if get_origin(anno) not in _SIMPLE_CONTAINERS:
            raise definitions.ChainsUsageError(error_msg)
        args = get_args(anno)
        for arg in args:
            if arg not in _SIMPLE_TYPES:
                raise definitions.ChainsUsageError(error_msg)
        return
    if utils.issubclass_safe(anno, pydantic.BaseModel):
        return

    raise definitions.ChainsUsageError(error_msg)


def _validate_endpoint_params(
    params: list[inspect.Parameter], cls_name: str
) -> list[definitions.InputArg]:
    if len(params) == 0:
        raise definitions.ChainsUsageError(
            f"`{cls_name}.{definitions.ENDPOINT_METHOD_NAME}` must be a method, i.e. "
            f"with `{definitions.SELF_ARG_NAME}` argument."
        )
    if params[0].name != definitions.SELF_ARG_NAME:
        raise definitions.ChainsUsageError(
            f"`{cls_name}.{definitions.ENDPOINT_METHOD_NAME}` must be a method, i.e. "
            f"with `{definitions.SELF_ARG_NAME}` argument."
        )
    input_args = []
    for param in params[1:]:  # Skip self argument.
        if param.annotation == inspect.Parameter.empty:
            raise definitions.ChainsUsageError(
                "Inputs of endpoints must have type annotations. For "
                f"`{cls_name}.{definitions.ENDPOINT_METHOD_NAME}` parameter "
                f"`{param.name}` has no type annotation."
            )
        _validate_io_type(param)
        type_descriptor = definitions.TypeDescriptor(raw=param.annotation)
        is_optional = param.default != inspect.Parameter.empty
        input_args.append(
            definitions.InputArg(
                name=param.name, type=type_descriptor, is_optional=is_optional
            )
        )
    return input_args


def _validate_and_describe_endpoint(
    cls: Type[definitions.ABCChainlet],
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
        raise definitions.ChainsUsageError(
            f"`{cls.__name__}` must have a {definitions.ENDPOINT_METHOD_NAME}` method."
        )
    # This is the unbound method.
    endpoint_method = getattr(cls, definitions.ENDPOINT_METHOD_NAME)
    if not inspect.isfunction(endpoint_method):
        raise definitions.ChainsUsageError(
            f"`{cls.__name__}.{definitions.ENDPOINT_METHOD_NAME}` must be a method."
        )
    signature = inspect.signature(endpoint_method)
    input_args = _validate_endpoint_params(
        list(signature.parameters.values()), cls.__name__
    )
    if signature.return_annotation == inspect.Parameter.empty:
        raise definitions.ChainsUsageError(
            "Return values of endpoints must be type annotated. Got:\n"
            f"{cls.__name__}.{definitions.ENDPOINT_METHOD_NAME}{signature} -> !MISSING!"
        )
    if get_origin(signature.return_annotation) is tuple:
        output_types = list(
            definitions.TypeDescriptor(raw=arg)
            for arg in get_args(signature.return_annotation)
        )
    else:
        output_types = [definitions.TypeDescriptor(raw=signature.return_annotation)]

    if inspect.isasyncgenfunction(endpoint_method):
        is_async = True
        is_generator = True
    elif inspect.iscoroutinefunction(endpoint_method):
        is_async = True
        is_generator = False
    else:
        is_async = False
        is_generator = inspect.isgeneratorfunction(endpoint_method)

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


def _validate_dependency_arg(param) -> ChainletDependencyMarker:
    # TODO: handle subclasses, unions, optionals, check default value etc.
    if param.name == definitions.CONTEXT_ARG_NAME:
        raise definitions.ChainsUsageError(
            f"The init argument name `{definitions.CONTEXT_ARG_NAME}` is reserved for "
            "the optional context argument, which must be trailing if used. Example "
            "of correct `__init__` with context:\n"
            f"{_example_chainlet_code()}"
        )

    if not isinstance(param.default, ChainletDependencyMarker):
        raise definitions.ChainsUsageError(
            f"Any arguments of a chainlet's __init__ (besides `context`) must have "
            "dependency chainlets with default values from `chains.provide`-directive. "
            f"Got `{param}`.\n"
            f"Example of correct `__init__` with dependencies:\n"
            f"{_example_chainlet_code()}"
        )
    chainlet_cls = param.default.chainlet_cls
    if not utils.issubclass_safe(chainlet_cls, definitions.ABCChainlet):
        raise definitions.ChainsUsageError(
            f"`{chainlet_cls}` must be a subclass of `{definitions.ABCChainlet}`."
        )
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
        raise definitions.ChainsUsageError(
            f"The type annotation for `{param.name}` must either be a `{Protocol}` "
            "or a class/subclass of the Chainlet type used as default value. "
            f"Got `{param.annotation}`."
        )
    return param.default  # The Marker.


class _ChainletInitValidator:
    """The `__init__`-method of a Chainlet must have the following signature:

    ```
    def __init__(
        self,
        [dep_0: dep_0_type = truss_chains.provide(dep_0_class),]
        [dep_1: dep_1_type = truss_chains.provide(dep_1_class),]
        ...
        [dep_N: dep_N_type = truss_chains.provide(dep_N_class),]
        [context: truss_chains.Context[UserConfig] = truss_chains.provide_context()]
    ) -> None:
    ```
    * The context argument is optionally trailing and must have a default constructed
     with the  `provide_context` directive. The type can be templated by a user
     defined config e.g. `truss_chains.Context[UserConfig]`.
    * The names and number of chainlet "dependency" arguments are arbitrary.
    * Default values for dependencies must be constructed with the `provide` directive
      to make the dependency injection work. The argument to `provide` must be a
      Chainlet class.
    * The type annotation for dependencies can be a Chainlet class, but it can also be
      a `Protocol` with an equivalent `run` method (e.g. for getting correct type
      checks when providing fake Chainlets for local testing.). It may be omitted if
      the type is clear from the RHS.
    """

    has_context: bool
    validated_dependencies: Mapping[str, definitions.DependencyDescriptor]

    def __init__(self, cls: Type[definitions.ABCChainlet]) -> None:
        if not cls.has_custom_init():
            self.has_context = False
            self.validated_dependencies = {}
            return
        # Each validation pops of "processed" arguments from the list.
        params = list(inspect.signature(cls.__init__).parameters.values())
        params = self._validate_self_arg(list(params))
        params, self.has_context = self._validate_context_arg(params)
        self.validated_dependencies = self._validate_dependencies(params)

    @staticmethod
    def _validate_self_arg(params: list[inspect.Parameter]) -> list[inspect.Parameter]:
        if len(params) == 0:
            raise definitions.ChainsUsageError(
                "Methods must have first argument `self`, got no arguments."
            )
        param = params.pop(0)
        if param.name != definitions.SELF_ARG_NAME:
            raise definitions.ChainsUsageError(
                f"Methods must have first argument `self`, got `{param.name}`."
            )
        return params

    @staticmethod
    def _validate_context_arg(
        params: list[inspect.Parameter],
    ) -> tuple[list[inspect.Parameter], bool]:
        def make_context_exception():
            return definitions.ChainsUsageError(
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
            raise make_context_exception()

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
            raise make_context_exception()
        if not isinstance(param.default, ContextDependencyMarker):
            raise definitions.ChainsUsageError(
                f"Incorrect default value `{param.default}` for `context` argument. "
                "Example of correct `__init__` with dependencies:\n"
                f"{_example_chainlet_code()}"
            )

        return params, has_context

    @staticmethod
    def _validate_dependencies(
        params,
    ) -> Mapping[str, definitions.DependencyDescriptor]:
        used = set()
        dependencies = {}
        for param in params:
            marker = _validate_dependency_arg(param)
            if marker.chainlet_cls in used:
                raise definitions.ChainsUsageError(
                    f"The same Chainlet class cannot be used multiple times for "
                    f"different arguments. Got previously used "
                    f"`{marker.chainlet_cls}` for `{param.name}`."
                )
            dependencies[param.name] = definitions.DependencyDescriptor(
                chainlet_cls=marker.chainlet_cls, options=marker.options
            )
            used.add(marker)
        return dependencies


def _validate_chainlet_cls(cls: Type[definitions.ABCChainlet]) -> None:
    # TODO: ensure Chainlets are only accessed via `provided` in `__init__`,
    #   not from manual instantiations on module-level or nested in a Chainlet.
    #   See other constraints listed in:
    # https://www.notion.so/ml-infra/WIP-Orchestration-a8cb4dad00dd488191be374b469ffd0a?pvs=4#7df299eb008f467a80f7ee3c0eccf0f0
    if not hasattr(cls, definitions.REMOTE_CONFIG_NAME):
        raise definitions.ChainsUsageError(
            f"Chainlets must have a `{definitions.REMOTE_CONFIG_NAME}` class variable "
            f"`{definitions.REMOTE_CONFIG_NAME} = {definitions.RemoteConfig.__name__}"
            f"(...)`. Missing for `{cls}`."
        )
    if not isinstance(
        remote_config := getattr(cls, definitions.REMOTE_CONFIG_NAME),
        definitions.RemoteConfig,
    ):
        raise definitions.ChainsUsageError(
            f"Chainlets must have a `{definitions.REMOTE_CONFIG_NAME}` class variable "
            f"of type `{definitions.RemoteConfig}`. Got `{type(remote_config)}` "
            f"for `{cls}`."
        )


def check_and_register_class(cls: Type[definitions.ABCChainlet]) -> None:
    _validate_chainlet_cls(cls)

    init_validator = _ChainletInitValidator(cls)
    chainlet_descriptor = definitions.ChainletAPIDescriptor(
        chainlet_cls=cls,
        dependencies=init_validator.validated_dependencies,
        has_context=init_validator.has_context,
        endpoint=_validate_and_describe_endpoint(cls),
        src_path=os.path.abspath(inspect.getfile(cls)),
        user_config_type=definitions.TypeDescriptor(raw=type(cls.default_user_config)),
    )
    logging.debug(f"Descriptor for {cls}:\n{chainlet_descriptor}\n")
    global_chainlet_registry.register_chainlet(chainlet_descriptor)


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

    def register_chainlet(self, chainlet_descriptor: definitions.ChainletAPIDescriptor):
        for dep in chainlet_descriptor.dependencies.values():
            # To depend on a Chainlet, the class must be defined (module initialized)
            # which entails that is has already been added to the registry.
            if dep.chainlet_cls not in self._chainlets:
                logging.error(f"Available chainlets: {list(self._chainlets.keys())}")
                raise KeyError(dep.chainlet_cls)

        # Because class are globally unique, to prevent re-use / overwriting of names,
        # We must check this in addition.
        if chainlet_descriptor.name in self._name_to_cls:
            conflict = self._name_to_cls[chainlet_descriptor.name]
            existing_source_path = self._chainlets[conflict].src_path
            raise definitions.ChainsUsageError(
                f"A chainlet with name `{chainlet_descriptor.name}` was already "
                f"defined, chainlet names must be globally unique.\n"
                f"Pre-existing in: `{existing_source_path}`\n"
                f"New conflict in: `{chainlet_descriptor.src_path}`."
            )

        self._chainlets[chainlet_descriptor.chainlet_cls] = chainlet_descriptor
        self._name_to_cls[chainlet_descriptor.name] = chainlet_descriptor.chainlet_cls

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


global_chainlet_registry = _ChainletRegistry()


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
                    f"When initializing Chainlet `{cls.__name__}`, for context "
                    f"argument an incompatible value was passed, value: `{value}`."
                )
                raise definitions.ChainsRuntimeError(
                    _instantiation_error_msg(cls.__name__)
                )
        # The argument is a dependency chainlet.
        elif isinstance(value, _BaseProvisionMarker):
            logging.error(
                f"When initializing Chainlet `{cls.__name__}`, for dependency chainlet"
                f"argument `{name}` an incompatible value was passed, value: `{value}`."
            )
            raise definitions.ChainsRuntimeError(_instantiation_error_msg(cls.__name__))


# Local Deployment #####################################################################

# A variable to track the stack depth relative to `run_local` context manager.
run_local_stack_depth: contextvars.ContextVar[int] = contextvars.ContextVar(
    "run_local_stack_depth"
)


def _create_modified_init_for_local(
    chainlet_descriptor: definitions.ChainletAPIDescriptor,
    cls_to_instance: MutableMapping[
        Type[definitions.ABCChainlet], definitions.ABCChainlet
    ],
    secrets: Mapping[str, str],
    data_dir: Optional[pathlib.Path],
    chainlet_to_service: Mapping[str, definitions.ServiceDescriptor],
    user_env: Mapping[str, str],
):
    """Replaces the default argument values with local Chainlet instantiations.

    If this patch is used, Chainlets can be functionally instantiated without
    any init args (because the patched defaults are sufficient).
    """

    def _verify_stack(stack: list[inspect.FrameInfo], levels_below_run_local: int):
        # TODO: this checks is incompatible with sub-classing chainlets.
        for frame in stack[:levels_below_run_local]:
            # This is a robust way to compare for function identity, since `wraps`
            # actually changes the name.
            if frame.frame.f_code != __init_local__.__code__:  # type: ignore[attr-defined]
                assert frame.code_context is not None
                logging.error(
                    f"Chainlet init called outside {__init_local__.__name__}, "
                    f'occurred in:\n File "{frame.filename}", line {frame.lineno}, in '
                    f"{frame.function}\n  {frame.code_context[0].strip()}."
                )
                raise definitions.ChainsRuntimeError(
                    _instantiation_error_msg(chainlet_descriptor.name)
                )

    original_init = chainlet_descriptor.chainlet_cls.__init__

    @functools.wraps(original_init)
    def __init_local__(self: definitions.ABCChainlet, **kwargs) -> None:
        logging.debug(f"Patched `__init__` of `{chainlet_descriptor.name}`.")
        stack_depth = run_local_stack_depth.get(None)
        assert stack_depth is not None, "The patched init is only called in context."
        stack = inspect.stack()
        current_stack_depth = len(stack)
        levels_below_run_local = current_stack_depth - stack_depth
        _verify_stack(stack, levels_below_run_local)
        kwargs_mod = dict(kwargs)
        if (
            chainlet_descriptor.has_context
            and definitions.CONTEXT_ARG_NAME not in kwargs_mod
        ):
            kwargs_mod[definitions.CONTEXT_ARG_NAME] = definitions.DeploymentContext(
                user_config=chainlet_descriptor.chainlet_cls.default_user_config,
                secrets=secrets,
                data_dir=data_dir,
                chainlet_to_service=chainlet_to_service,
                user_env=user_env,
            )
        else:
            logging.debug(
                f"Use explicitly given context for `{self.__class__.__name__}`."
            )
        for arg_name, dep in chainlet_descriptor.dependencies.items():
            chainlet_cls = dep.chainlet_cls
            if arg_name in kwargs_mod:
                logging.debug(
                    f"Use explicitly given instance for `{arg_name}` "
                    f"of type `{dep.name}`."
                )
                continue
            if chainlet_cls in cls_to_instance:
                logging.debug(
                    f"Use previously created instance for `{arg_name}` "
                    f"of type `{dep.name}`."
                )
                instance = cls_to_instance[chainlet_cls]
            else:
                logging.debug(
                    f"Create new instance for `{arg_name}` of type `{dep.name}`. "
                    f"Calling patched __init__."
                )
                assert chainlet_cls._init_is_patched
                # Dependency chainlets are instantiated here, using their __init__
                # that is patched for local.
                instance = chainlet_cls()  # type: ignore  # Here init args are patched.
                cls_to_instance[chainlet_cls] = instance

            kwargs_mod[arg_name] = instance

        logging.debug(f"Calling original __init__ of {chainlet_descriptor.name}.")
        original_init(self, **kwargs_mod)

    return __init_local__


@contextlib.contextmanager
def run_local(
    secrets: Mapping[str, str],
    data_dir: Optional[pathlib.Path],
    chainlet_to_service: Mapping[str, definitions.ServiceDescriptor],
    user_env: Mapping[str, str],
) -> Any:
    """Context to run Chainlets with dependency injection from local instances."""
    # TODO: support retries in local mode.
    type_to_instance: MutableMapping[
        Type[definitions.ABCChainlet], definitions.ABCChainlet
    ] = {}
    original_inits: MutableMapping[Type[definitions.ABCChainlet], Callable] = {}

    # Capture the stack depth when entering the context manager
    stack_depth = len(inspect.stack())
    token = None
    for chainlet_descriptor in global_chainlet_registry.chainlet_descriptors:
        original_inits[chainlet_descriptor.chainlet_cls] = (
            chainlet_descriptor.chainlet_cls.__init__
        )
        init_for_local = _create_modified_init_for_local(
            chainlet_descriptor,
            type_to_instance,
            secrets,
            data_dir,
            chainlet_to_service,
            user_env,
        )
        chainlet_descriptor.chainlet_cls.__init__ = init_for_local  # type: ignore[method-assign]
        chainlet_descriptor.chainlet_cls._init_is_patched = True
    try:
        # Subtract 2 levels: `run_local` (this) and `__enter__` (from @contextmanager).
        token = run_local_stack_depth.set(stack_depth - 2)
        yield
    finally:
        # Restore original classes to unpatched state.
        for chainlet_cls, original_init in original_inits.items():
            chainlet_cls.__init__ = original_init  # type: ignore[method-assign]
            chainlet_cls._init_is_patched = False
        if token is not None:
            run_local_stack_depth.reset(token)


########################################################################################


def entrypoint(cls: Type[ChainletT]) -> Type[ChainletT]:
    """Decorator to tag a chainlet as an entrypoint."""
    if not (utils.issubclass_safe(cls, definitions.ABCChainlet)):
        raise definitions.ChainsUsageError(
            "Only chainlet classes can be marked as entrypoint."
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
    module_path = pathlib.Path(module_path).resolve()
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
    module.__file__ = str(module_path)
    # Since the framework depends on tracking the source files via `inspect` and this
    # depends on the modules bein properly registered in `sys.modules`, we have to
    # manually do this here (because importlib does not do it automatically). This
    # registration has to stay at least until the deployment command has finished.
    if module_name in sys.modules:
        raise ImportError(
            f"{import_error_msg}. There is already a module in `sys.modules` "
            f"with name `{module_name}`. Overwriting that value is unsafe. "
            "Try renaming your source file."
        )
    modules_before = set(sys.modules.keys())
    sys.modules[module_name] = module
    # Add path for making absolute imports relative to the source_module's dir.
    sys.path.insert(0, str(module_path.parent))
    try:
        spec.loader.exec_module(module)

        if target_name:
            target_cls = getattr(module, target_name, None)
            if not target_cls:
                raise AttributeError(
                    f"Target chainlet class `{target_name}` not found in `{module_path}`."
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
                    f"No `target_name` was specified and no chainlet in `{module_path}` "
                    "was tagged with `@chains.mark_entrypoint`. Tag one chainlet or provide "
                    "the chainlet class name."
                )
            elif len(entrypoints) > 1:
                raise ValueError(
                    "`target_name` was not specified and multiple chainlets in "
                    f"`{module_path}` were tagged with `@chains.mark_entrypoint`. Tag one "
                    "chainlet or provide the chainlet class name. Found chainlets: \n"
                    f"{entrypoints}"
                )
            target_cls = utils.expect_one(entrypoints)
            if not utils.issubclass_safe(target_cls, definitions.ABCChainlet):
                raise TypeError(
                    f"Target `{target_cls}` is not a {definitions.ABCChainlet}."
                )

        yield target_cls

    finally:
        modules_to_delete = set(sys.modules.keys()) - modules_before
        logging.debug(
            f"Deleting modules when exiting import context: {modules_to_delete}"
        )
        for mod in modules_to_delete:
            del sys.modules[mod]
        try:
            sys.path.remove(str(module_path.parent))
        except ValueError:  # In case the value was already removed for whatever reason.
            pass
