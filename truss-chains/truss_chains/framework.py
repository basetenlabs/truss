import collections
import contextlib
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
    Mapping,
    MutableMapping,
    Optional,
    Protocol,
    Type,
    get_args,
    get_origin,
)

import pydantic
from truss_chains import definitions

_SIMPLE_TYPES = {int, float, complex, bool, str, bytes, None}
_SIMPLE_CONTAINERS = {list, dict}

_DOCS_URL_CHAINING = "https://docs.baseten.co/chains/chaining-chainlets"
_DOCS_URL_LOCAL = "https://docs.baseten.co/chains/gettin-started"


# Checking of Chainlet class definition ###############################################

# TODO: make this a code template to have type checking on it.
_EXAMPLE_CHAINLET = """
class Example(chains.ChainletBase):
    remote_config = chains.RemoteConfig(docker_image=...)

    def __init__(
        self,
        context: chains.DeploymentContext = chains.provide_context(),
        data_generator: GenerateData = chains.provide(GenerateData),
        splitter: shared_chainlet.SplitText = chains.provide(shared_chainlet.SplitText),
    ) -> None:
        super().__init__(context)
        self._data_generator = data_generator
        self._data_splitter = splitter
"""


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
        f"{_EXAMPLE_CHAINLET}"
    )


def _validate_io_type(param: inspect.Parameter) -> None:
    """
    For Chainlet I/O (both data or parameters), we allow simple types
    (int, str, float...) and `list` or `dict` containers of these.
    Any deeper nested and structured data must be typed as a pydantic model.
    """
    anno = param.annotation
    if isinstance(anno, str):
        raise definitions.APIDefinitionError(
            f"A string-valued type annotation was found: `{param}`. Use only actual "
            "types and avoid `from __future__ import annotations` (upgrade python)."
        )
    if anno in _SIMPLE_TYPES:
        return
    if isinstance(anno, types.GenericAlias):
        if get_origin(anno) not in _SIMPLE_CONTAINERS:
            raise definitions.APIDefinitionError(
                f"For generic types, only containers {_SIMPLE_CONTAINERS} are "
                f"allowed, but got `{param}`."
            )
        args = get_args(anno)
        for arg in args:
            if arg not in _SIMPLE_TYPES:
                raise definitions.APIDefinitionError(
                    f"For generic types, only the following item types {_SIMPLE_TYPES} "
                    f"are allowed, but got `{param}`."
                )
        return
    if issubclass(anno, pydantic.BaseModel):
        # TODO: generating models for the stubs only covers the schema-capabilities
        #   but not any methods that clients might add to their pydantic models.
        #   We should enforce that no user-defined methods (or class vars or properties)
        #   are present, since this might lead to broken behavior.
        # TODO: for enums we rely on the convention that they are string enums and
        #  the member names are the capitalized member values (or better the values are
        #  capitalized themselves too). This constraint should be enforced / checked.
        #  We could also require using a custom StrEnum-like enum base class.
        try:
            anno.schema()
        except Exception as e:
            raise definitions.APIDefinitionError(
                "Pydantic models used for I/O typs must be serializable and able to "
                f"generate a schema. Please fix `{param}`."
            ) from e
        return

    raise definitions.APIDefinitionError(
        f"Unsupported I/O type `{param}`. Use simple types {_SIMPLE_TYPES}, "
        f"containers ({_SIMPLE_CONTAINERS}) of simple types or serializable pydantic "
        "models."
    )


def _validate_endpoint_params(
    params: list[inspect.Parameter], cls_name: str
) -> list[tuple[str, definitions.TypeDescriptor]]:
    if len(params) == 0:
        raise definitions.APIDefinitionError(
            f"`{cls_name}.{definitions.ENDPOINT_METHOD_NAME}` must be a method, i.e. "
            f"with `{definitions.SELF_ARG_NAME}` argument."
        )
    if params[0].name != definitions.SELF_ARG_NAME:
        raise definitions.APIDefinitionError(
            f"`{cls_name}.{definitions.ENDPOINT_METHOD_NAME}` must be a method, i.e. "
            f"with `{definitions.SELF_ARG_NAME}` argument."
        )
    input_name_and_types = []
    for param in params[1:]:  # Skip self argument.
        if param.annotation == inspect.Parameter.empty:
            raise definitions.APIDefinitionError(
                "Inputs of endpoints must have type annotations. For "
                f"`{cls_name}.{definitions.ENDPOINT_METHOD_NAME}` parameter "
                f"`{param.name}` has not type annotation."
            )
        _validate_io_type(param)
        type_descriptor = definitions.TypeDescriptor(raw=param.annotation)
        input_name_and_types.append((param.name, type_descriptor))
    return input_name_and_types


def _validate_and_describe_endpoint(
    cls: Type[definitions.ABCChainlet],
) -> definitions.EndpointAPIDescriptor:
    """The "endpoint method" of a Chainlet must have the following signature:

    ```
    [async] def run(
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
        raise definitions.APIDefinitionError(
            f"`{cls.__name__}` must have a {definitions.ENDPOINT_METHOD_NAME}` method."
        )
    # This is the unbound method.
    endpoint_method = getattr(cls, definitions.ENDPOINT_METHOD_NAME)
    if not inspect.isfunction(endpoint_method):
        raise definitions.APIDefinitionError(
            f"`{cls.__name__}.{definitions.ENDPOINT_METHOD_NAME}` must be a method."
        )
    signature = inspect.signature(endpoint_method)
    input_name_and_types = _validate_endpoint_params(
        list(signature.parameters.values()), cls.__name__
    )
    if signature.return_annotation == inspect.Parameter.empty:
        raise definitions.APIDefinitionError(
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
        input_names_and_types=input_name_and_types,
        output_types=output_types,
        is_async=is_async,
        is_generator=is_generator,
    )


def _get_generic_class_type(var):
    """Extracts `SomeGeneric` from `SomeGeneric` or `SomeGeneric[T]` uniformly."""
    origin = get_origin(var)
    return origin if origin is not None else var


def _validate_dependency_arg(param) -> Type[definitions.ABCChainlet]:
    # TODO: handle subclasses, unions, optionals, check default value etc.
    if not isinstance(param.default, ChainletProvisionPlaceholder):
        raise definitions.APIDefinitionError(
            f"Any arguments of a chainlet's __init__ (besides `context`) must have"
            "dependency chainlets with default values from `chains.provide`-directive. "
            f"Got `{param}`.\n"
            f"Example of correct `__init__` with dependencies:\n"
            f"{_EXAMPLE_CHAINLET}"
        )
    chainlet_cls = param.default.chainlet_cls
    if not (
        # TODO: `Protocol` is not a proper class and this might be version dependent.
        # Find a better way to inspect this.
        issubclass(param.annotation, Protocol)  # type: ignore[arg-type]
        or issubclass(chainlet_cls, param.annotation)
    ):
        raise definitions.APIDefinitionError(
            f"The type annotation for `{param.name}` must either be a `{Protocol}` "
            "or a class/subclass of the Chainlet type used as default value. "
            f"Got `{param.default}`."
        )
    if not issubclass(chainlet_cls, definitions.ABCChainlet):
        raise definitions.APIDefinitionError(
            f"`{chainlet_cls}` must be a subclass of `{definitions.ABCChainlet}`."
        )
    return chainlet_cls


class _ChainletInitParams:
    def __init__(self, params: list[inspect.Parameter]) -> None:
        self._params = params
        self._validate_self_arg()
        self._validate_context_arg()

    def _validate_self_arg(self) -> None:
        if len(self._params) == 0:
            raise definitions.APIDefinitionError(
                "Methods must have first argument `self`."
            )

        if self._params[0].name != definitions.SELF_ARG_NAME:
            raise definitions.APIDefinitionError(
                "Methods must have first argument `self`."
            )

    def _validate_context_arg(self) -> None:
        context_exception = definitions.APIDefinitionError(
            f"`{definitions.ABCChainlet}` must have "
            f"`{definitions.CONTEXT_ARG_NAME}` argument of type "
            f"`{definitions.DeploymentContext}`.\n"
            f"Got: `{self._params}`.\n"
            "Example of correct `__init__` with dependencies:\n"
            f"{_EXAMPLE_CHAINLET}"
        )
        if len(self._params) < 2:
            raise context_exception
        if self._params[1].name != definitions.CONTEXT_ARG_NAME:
            raise context_exception

        param = self._params[1]
        param_type = _get_generic_class_type(param.annotation)
        # We are lenient and allow omitting the type annotation for context...
        if (
            (param_type is not None)
            and (param_type != inspect.Parameter.empty)
            and (not issubclass(param_type, definitions.DeploymentContext))
        ):
            print(param_type)
            raise context_exception
        if not isinstance(param.default, ContextProvisionPlaceholder):
            raise definitions.APIDefinitionError(
                f"Incorrect default value `{param.default}` for `context` argument. "
                "Example of correct `__init__` with dependencies:\n"
                f"{_EXAMPLE_CHAINLET}"
            )

    def validated_dependencies(self) -> Mapping[str, Type[definitions.ABCChainlet]]:
        used_classes = set()
        dependencies = {}
        for param in self._params[2:]:  # Skip self and context.
            chainlet_cls = _validate_dependency_arg(param)
            if chainlet_cls in used_classes:
                raise definitions.APIDefinitionError(
                    f"The same Chainlet class cannot be used multiple times for "
                    f"different arguments. Got previously used `{chainlet_cls}` "
                    f"for `{param.name}`."
                )
            dependencies[param.name] = chainlet_cls
            used_classes.add(chainlet_cls)
        return dependencies


def _validate_init_and_get_dependencies(
    cls: Type[definitions.ABCChainlet],
) -> Mapping[str, Type[definitions.ABCChainlet]]:
    """The `__init__`-method of a Chainlet must have the following signature:

    ```
    def __init__(
        self,
        context: truss_chains.Context = truss_chains.provide_context(),
        [dep_0: dep_0_type = truss_chains.provide(dep_0_proc_class),]
        [dep_1: dep_1_type = truss_chains.provide(dep_1_proc_class),]
        ...
    ) -> None:
    ```

    * The context argument is required and must have a default constructed with the
      `provide_context` directive. The type can be templated by a user defined config
      e.g. `truss_chains.Context[UserConfig]`.
    * The names and number of other - "dependency" - arguments are arbitrary.
    * Default values for dependencies must be constructed with the `provide` directive
      to make the dependency injection work. The argument to `provide` must be a
      Chainlet class.
    * The type annotation for dependencies can be a Chainlet class, but it can also be
      a `Protocol` with an equivalent `run` method (e.g. for getting correct type
      checks when providing fake Chainlets for local testing.).
    """
    params = _ChainletInitParams(
        list(inspect.signature(cls.__init__).parameters.values())
    )
    return params.validated_dependencies()


def _validate_chainlet_cls(cls: Type[definitions.ABCChainlet]) -> None:
    # TODO: ensure Chainlets are only accessed via `provided` in `__init__`,
    #   not from manual instantiations on module-level or nested in a Chainlet.
    #   See other constraints listed in:
    # https://www.notion.so/ml-infra/WIP-Orchestration-a8cb4dad00dd488191be374b469ffd0a?pvs=4#7df299eb008f467a80f7ee3c0eccf0f0
    if not hasattr(cls, definitions.REMOTE_CONFIG_NAME):
        raise definitions.APIDefinitionError(
            f"Chainlets must have a `{definitions.REMOTE_CONFIG_NAME}` class variable "
            f"`{definitions.REMOTE_CONFIG_NAME} = {definitions.RemoteConfig.__name__}"
            f"(...)`. Missing for `{cls}`."
        )
    if not isinstance(
        remote_config := getattr(cls, definitions.REMOTE_CONFIG_NAME),
        definitions.RemoteConfig,
    ):
        raise definitions.APIDefinitionError(
            f"Chainlets must have a `{definitions.REMOTE_CONFIG_NAME}` class variable "
            f"of type `{definitions.RemoteConfig}`. Got `{type(remote_config)}` "
            f"for `{cls}`."
        )


def check_and_register_class(cls: Type[definitions.ABCChainlet]) -> None:
    _validate_chainlet_cls(cls)
    chainlet_descriptor = definitions.ChainletAPIDescriptor(
        chainlet_cls=cls,
        dependencies=_validate_init_and_get_dependencies(cls),
        endpoint=_validate_and_describe_endpoint(cls),
        src_path=os.path.abspath(inspect.getfile(cls)),
        user_config_type=definitions.TypeDescriptor(raw=type(cls.default_user_config)),
    )
    logging.debug(f"Descriptor for {cls}:\n{chainlet_descriptor}\n")
    global_chainlet_registry.register_chainlet(chainlet_descriptor)


# Dependency-Injection / Registry ######################################################


class _BaseProvisionPlaceholder:
    """A marker for object to be dependency injected by the framework."""


class ChainletProvisionPlaceholder(_BaseProvisionPlaceholder):
    chainlet_cls: Type[definitions.ABCChainlet]

    def __init__(self, chainlet_cls: Type[definitions.ABCChainlet]) -> None:
        self.chainlet_cls = chainlet_cls

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.chainlet_cls.__name__})"

    def __getattr__(self, item: str) -> Any:
        if item == definitions.ENDPOINT_METHOD_NAME:
            logging.error(f"Attempting to access attribute `{item}` on `{self}`.")
            raise definitions.UsageError(
                f"It seems `chains.provide({self.chainlet_cls.__name__})` was used "
                f"outside the `__init__` method of a chainlet. Dependency chainlets "
                f"must be passed as init arguments.\n"
                f"See {_DOCS_URL_CHAINING}.\n"
                "Example of correct `__init__` with dependencies:\n"
                f"{_EXAMPLE_CHAINLET}"
            )


class ContextProvisionPlaceholder(_BaseProvisionPlaceholder):
    def __str__(self) -> str:
        return f"{self.__class__.__name__}"


class _ChainletRegistry:
    # Because dependencies are required to be present when registering a Chainlet,
    # this dict contains natively a topological sorting of the dependency graph.
    _chainlets: collections.OrderedDict[
        Type[definitions.ABCChainlet], definitions.ChainletAPIDescriptor
    ]
    _name_to_cls: MutableMapping[str, Type]

    def __init__(self) -> None:
        self._chainlets = collections.OrderedDict()
        self._name_to_cls = {}

    def register_chainlet(self, chainlet_descriptor: definitions.ChainletAPIDescriptor):
        for dep in chainlet_descriptor.dependencies.values():
            # To depend on a Chainlet, the class must be defined (module initialized)
            # which entails that is has already been added to the registry.
            assert dep in self._chainlets, dep

        # Because class are globally unique, to prevent re-use / overwriting of names,
        # We must check this in addition.
        if chainlet_descriptor.name in self._name_to_cls:
            conflict = self._name_to_cls[chainlet_descriptor.name]
            existing_source_path = self._chainlets[conflict].src_path
            raise definitions.APIDefinitionError(
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
        self, Chainlet: definitions.ChainletAPIDescriptor
    ) -> Iterable[definitions.ChainletAPIDescriptor]:
        return [
            self._chainlets[desc]
            for desc in self._chainlets[Chainlet.chainlet_cls].dependencies.values()
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
    """Asserts all placeholder markers are replaced by actual objects."""
    final_args = _determine_arguments(original_init, **kwargs)
    for name, value in final_args.items():
        if isinstance(value, _BaseProvisionPlaceholder):
            logging.error(
                f"When initializing class `{cls.__name__}`, for "
                f"default argument `{name}` a symbolic placeholder value "
                f"was passed (`{value}`)."
            )
            raise definitions.UsageError(_instantiation_error_msg(cls.__name__))


# Local Deployment #####################################################################


def _create_local_context(
    chainlet_cls: Type[definitions.ABCChainlet], secrets: Mapping[str, str]
) -> definitions.DeploymentContext:
    return definitions.DeploymentContext(
        user_config=chainlet_cls.default_user_config, secrets=secrets
    )


def _create_modified_init_for_local(
    chainlet_descriptor: definitions.ChainletAPIDescriptor,
    cls_to_instance: MutableMapping[
        Type[definitions.ABCChainlet], definitions.ABCChainlet
    ],
    secrets: Mapping[str, str],
):
    """Replaces the default argument values with local Chainlet instantiations.

    If this patch is used, Chainlets can be functionally instantiated without
    any init args (because the patched defaults are sufficient).
    """
    # TODO: can we somehow detect if a chainlet is instantiated inside another chainlet
    #  (instead of init), e.g. users trying to instantiated a chainlet inside
    #  `run` or `__init__`? Problem: if code is run with `run_local` context,
    #  all chainlet classes are patched, so this will not raise an error - though
    #  in the remote deployment it would trigger the actionable instantiation message.
    original_init = chainlet_descriptor.chainlet_cls.__init__

    def init_for_local(self: definitions.ABCChainlet, **kwargs) -> None:
        logging.debug(f"Patched `__init__` of `{chainlet_descriptor.name}`.")
        kwargs_mod = dict(kwargs)
        if definitions.CONTEXT_ARG_NAME not in kwargs_mod:
            context = _create_local_context(chainlet_descriptor.chainlet_cls, secrets)
            kwargs_mod[definitions.CONTEXT_ARG_NAME] = context
        else:
            logging.debug(
                f"Use explicitly given context for `{self.__class__.__name__}`."
            )
        for arg_name, dep_cls in chainlet_descriptor.dependencies.items():
            if arg_name in kwargs_mod:
                logging.debug(
                    f"Use explicitly given instance for `{arg_name}` of "
                    f"type `{dep_cls.__name__}`."
                )
                continue
            if dep_cls in cls_to_instance:
                logging.debug(
                    f"Use previously created instance for `{arg_name}` of type "
                    f"`{dep_cls.__name__}`."
                )
                instance = cls_to_instance[dep_cls]
            else:
                logging.debug(
                    f"Create new instance for `{arg_name}` "
                    f"of type `{dep_cls.__name__}`."
                )
                assert dep_cls._init_is_patched
                instance = dep_cls()  # type: ignore  # Here init args are patched.
                cls_to_instance[dep_cls] = instance

            kwargs_mod[arg_name] = instance

        original_init(self, **kwargs_mod)

    return init_for_local


@contextlib.contextmanager
def run_local(secrets: Optional[Mapping[str, str]] = None) -> Any:
    """Context to run Chainlets with dependency injection from local instances."""
    type_to_instance: MutableMapping[
        Type[definitions.ABCChainlet], definitions.ABCChainlet
    ] = {}
    original_inits: MutableMapping[Type[definitions.ABCChainlet], Callable] = {}

    for chainlet_descriptor in global_chainlet_registry.chainlet_descriptors:
        original_inits[
            chainlet_descriptor.chainlet_cls
        ] = chainlet_descriptor.chainlet_cls.__init__
        init_for_local = _create_modified_init_for_local(
            chainlet_descriptor, type_to_instance, secrets or {}
        )
        chainlet_descriptor.chainlet_cls.__init__ = init_for_local  # type: ignore[method-assign]
        chainlet_descriptor.chainlet_cls._init_is_patched = True
    try:
        yield
    finally:
        # Restore original classes to unpatched state.
        for chainlet_cls, original_init in original_inits.items():
            chainlet_cls.__init__ = original_init  # type: ignore[method-assign]
            chainlet_cls._init_is_patched = False


########################################################################################


def import_target(
    module_path: pathlib.Path, target_name: str
) -> Type[definitions.ABCChainlet]:
    module_path = pathlib.Path(module_path).resolve()
    module_name = module_path.stem  # Use the file's name as the module name

    error_msg = f"Could not import `{target_name}` from `{module_path}`. Check path."
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if not spec:
        raise ImportError(error_msg)
    if not spec.loader:
        raise ImportError(error_msg)

    module = importlib.util.module_from_spec(spec)
    module.__file__ = str(module_path)
    # Since the framework depends on tracking the source files via `inspect` and this
    # depends on the modules bein properly registered in `sys.modules`, we have to
    # manually do this here (because importlib does not do it automatically). This
    # registration has to stay at least until the deployment command has finished.
    # TODO: Might need a context manager to remote this after deploy.
    sys.modules[module_name] = module
    # Add path for making absolute import w.r.t to the source_module's dir.
    sys.path.insert(0, str(module_path.parent))
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        del sys.modules[module_name]
        raise e from e
    finally:
        sys.path.pop(0)

    target_cls = getattr(module, target_name, None)
    if not target_cls:
        raise AttributeError(
            f"Target chainlet `{target_name}` not found in `{module_path}`."
        )
    if not issubclass(target_cls, definitions.ABCChainlet):
        raise TypeError(f"Target `{target_cls}` is not a {definitions.ABCChainlet}")

    return target_cls
