import collections
import contextlib
import inspect
import logging
import os
import pathlib
import shutil
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
from slay import code_gen, definitions, utils
from slay.truss_adapter import deploy

_SIMPLE_TYPES = {int, float, complex, bool, str, bytes, None}
_SIMPLE_CONTAINERS = {list, dict}


# Checking of processor class definition ###############################################


def _validate_io_type(param: inspect.Parameter) -> None:
    """
    For processor I/O (both data or parameters) we allow simple types
    (int, str, float...) and `list` or `dict` containers of these.
    Any deeper nested and structured data must be typed as a pydnatic model.
    """
    anno = param.annotation
    if anno in _SIMPLE_TYPES:
        return
    if isinstance(anno, types.GenericAlias):
        if get_origin(anno) not in _SIMPLE_CONTAINERS:
            raise definitions.APIDefinitonError(
                f"For generic types, only containers {_SIMPLE_CONTAINERS} are "
                f"allowed, but got `{param}`."
            )
        args = get_args(anno)
        for arg in args:
            if arg not in _SIMPLE_TYPES:
                raise definitions.APIDefinitonError(
                    f"For generic types, only arg types {_SIMPLE_TYPES} are "
                    f"allowed, but got `{param}`."
                )
        return
    if issubclass(anno, pydantic.BaseModel):
        try:
            anno.schema()
        except Exception as e:
            raise definitions.APIDefinitonError(
                "Pydantic annotations must be able to generate a schema. "
                f"Please fix `{param}`."
            ) from e
        return

    raise definitions.APIDefinitonError(anno)


def _validate_endpoint_params(
    params: list[inspect.Parameter], cls_name: str
) -> list[tuple[str, definitions.TypeDescriptor]]:
    if len(params) == 0:
        raise definitions.APIDefinitonError(
            f"`{cls_name}.{definitions.ENDPOINT_NAME}` must be a method, i.e. "
            "with `self` argument."
        )
    if params[0].name != definitions.SELF_ARG_NAME:
        raise definitions.APIDefinitonError(
            f"`{cls_name}.{definitions.ENDPOINT_NAME}` must be a method, i.e. "
            "with `self` argument."
        )
    input_name_and_types = []
    for param in params[1:]:  # Skip self argument.
        if param.annotation == inspect.Parameter.empty:
            raise definitions.APIDefinitonError(
                "Inputs of endpoints must have type annotations. "
                f"For `{cls_name}` got:\n{param}"
            )
        _validate_io_type(param)
        type_descriptor = definitions.TypeDescriptor(raw=param.annotation)
        input_name_and_types.append((param.name, type_descriptor))
    return input_name_and_types


def _validate_and_describe_endpoint(
    cls: Type[definitions.ABCProcessor],
) -> definitions.EndpointAPIDescriptor:
    """The "endpoint method" of a processor must have the follwing signature:

    ```
    [async] def run(
        self, [param_0: anno_0, param_1: anno_1 = default_1, ...]) -> ret_anno:
    ```

    * The name must be `run`.
    * It can be sync or sync or def.
    * The number and names of parameters arbitrary, both args and kwargs are ok.
    * All parameters and the return value must have type annotations. See
      `_validate_io_type` for valid types.
    * Generators are allowed, too (but not yet supported).
    """
    if not hasattr(cls, definitions.ENDPOINT_NAME):
        raise definitions.APIDefinitonError(
            f"`{cls.__name__}` must have a {definitions.ENDPOINT_NAME}` method."
        )
    endpoint_method = getattr(
        cls, definitions.ENDPOINT_NAME
    )  # This is the unbound method.
    if not inspect.isfunction(endpoint_method):
        raise definitions.APIDefinitonError(
            f"`{cls.__name__}.{definitions.ENDPOINT_NAME}` must be a method."
        )

    signature = inspect.signature(endpoint_method)
    input_name_and_types = _validate_endpoint_params(
        list(signature.parameters.values()), cls.__name__
    )

    if signature.return_annotation == inspect.Parameter.empty:
        raise definitions.APIDefinitonError(
            f"Return values of endpoints must be type annotated. Got:\n{signature}"
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
        input_names_and_tyes=input_name_and_types,
        output_types=output_types,
        is_async=is_async,
        is_generator=is_generator,
    )


def _get_generic_class_type(var):
    """Extracts `SomeGeneric` from `SomeGeneric` or `SomeGeneric[T]` uniformly."""
    origin = get_origin(var)
    return origin if origin is not None else var


def _validate_depenency_arg(param) -> Type[definitions.ABCProcessor]:
    # TODO: handle subclasses, unions, optionals, check default value etc.
    if not isinstance(param.default, ProcessorProvisionPlaceholder):
        raise definitions.APIDefinitonError(
            f"Any extra arguments of a processor's __init__ must have a default "
            f"value of type `{ProcessorProvisionPlaceholder}` (created with the "
            f"`provide` directive). Got `{param.default}` for `{param.name}`."
        )
    processor_cls = param.default.processor_cls
    if not (
        issubclass(param.annotation, Protocol)
        or issubclass(processor_cls, param.annotation)
    ):
        definitions.APIDefinitonError(
            f"The type annotaiton for `{param.name}` must either be a `{Protocol}` "
            "or a class/subclass of the processor type used as default value. "
            f"Got `{param.default}`."
        )
    if not issubclass(processor_cls, definitions.ABCProcessor):
        raise definitions.APIDefinitonError(
            f"`{processor_cls}` must be a subclass of `{definitions.ABCProcessor}`."
        )
    return processor_cls


class _ProcessorInitParams:
    def __init__(self, params: list[inspect.Parameter]) -> None:
        self._params = params
        self._validate_self_arg()
        self._validate_context_arg()

    def _validate_self_arg(self) -> None:
        if len(self._params) == 0:
            raise definitions.APIDefinitonError(
                "Methods must have first argument `self`."
            )

        if self._params[0].name != definitions.SELF_ARG_NAME:
            raise definitions.APIDefinitonError(
                "Methods must have first argument `self`."
            )

    def _validate_context_arg(self) -> None:
        context_exception = definitions.APIDefinitonError(
            f"`{definitions.ABCProcessor}` must have "
            f"`{definitions.CONTEXT_ARG_NAME}` argument of type "
            f"`{definitions.Context}`."
        )
        if len(self._params) < 2:
            raise context_exception
        if self._params[1].name != definitions.CONTEXT_ARG_NAME:
            raise context_exception

        param = self._params[1]
        param_type = _get_generic_class_type(param.annotation)
        if not issubclass(param_type, definitions.Context):
            raise context_exception
        if not isinstance(param.default, ContextProvisionPlaceholder):
            raise definitions.APIDefinitonError(
                f"The default value for the `context` argument of a processor's "
                f"__init__ must be of type `{ContextProvisionPlaceholder}` (created "
                f"with the `provide_context` directive). Got `{param.default}`."
            )

    def validated_dependencies(self) -> Mapping[str, Type[definitions.ABCProcessor]]:
        used_classes = set()
        dependencies = {}
        for param in self._params[2:]:  # Skip self and context.
            processor_cls = _validate_depenency_arg(param)
            if processor_cls in used_classes:
                raise definitions.APIDefinitonError(
                    f"The same processor class cannot be used multiple times for "
                    f"different arguments. Got previously used `{processor_cls}` "
                    f"for `{param.name}`."
                )
            dependencies[param.name] = processor_cls
            used_classes.add(processor_cls)
        return dependencies


def _validate_init_and_get_dependencies(
    cls: Type[definitions.ABCProcessor],
) -> Mapping[str, Type[definitions.ABCProcessor]]:
    """The `__init__`-method of a processor must have the follwing signature:
    ```
    def __init__(
        self,
        context: slay.Context = slay.provide_context(),
        [dep_0: dep_0_type = slay.provide(dep_0_proc_class),]
        [dep_1: dep_1_type = slay.provide(dep_1_proc_class),]
        ...
    ) -> None:

    * The context argument is required and must have a default construced with the
      `provide_context` directive. The type can be templated by a user defined config
      e.g. `slay.Context[UserConfig]`.
    * The names and number of other - "dependency" - arguments are arbitrary.
    * Default values for dependencies must be constructed with the `provide` directive
      to make the dependency injeciton work. The argument to `provide` must be a
      processor class.
    * The type annotation for depdencies can be a procssor class, but it can also be
      a `Protocol` with an equivalent `run` method (e.g. for getting correct type
      checks when providing fake processors for local testing.).
    """
    params = _ProcessorInitParams(
        list(inspect.signature(cls.__init__).parameters.values())
    )
    return params.validated_dependencies()


def _validate_variable_access(cls: Type[definitions.ABCProcessor]) -> None:
    # TODO ensure that processors are only accessed via `provided` in `__init__`,`
    # not from manual instantiations on module-level or nested in a processor.
    # See other constraints listed in:
    # https://www.notion.so/ml-infra/WIP-Orchestration-a8cb4dad00dd488191be374b469ffd0a?pvs=4#7df299eb008f467a80f7ee3c0eccf0f0
    ...


def check_and_register_class(cls: Type[definitions.ABCProcessor]) -> None:
    processor_descriptor = definitions.ProcessorAPIDescriptor(
        processor_cls=cls,
        depdendencies=_validate_init_and_get_dependencies(cls),
        endpoint=_validate_and_describe_endpoint(cls),
        src_path=os.path.abspath(inspect.getfile(cls)),
        user_config_type=definitions.TypeDescriptor(
            raw=type(cls.default_config.user_config)
        ),
    )
    logging.debug(f"Descriptor for {cls}:\n{processor_descriptor}\n")
    _validate_variable_access(cls)
    _global_processor_registry.register_processor(processor_descriptor)


# Dependency-Injection / Registry ######################################################


class _BaseProvisionPlaceholder:
    """A marker for object to be depdenency injected by the framework."""


class ProcessorProvisionPlaceholder(_BaseProvisionPlaceholder):
    # TODO: extend with RPC customization, e.g. timeouts, retries etc.
    processor_cls: Type[definitions.ABCProcessor]

    def __init__(self, processor_cls: Type[definitions.ABCProcessor]) -> None:
        self.processor_cls = processor_cls

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self._cls_name})"

    @property
    def _cls_name(self) -> str:
        return self.processor_cls.__name__


class ContextProvisionPlaceholder(_BaseProvisionPlaceholder):
    def __str__(self) -> str:
        return f"{self.__class__.__name__}"


class _ProcessorRegistry:
    # Because dependencies are required to be present when registering a processor,
    # this dict contains natively a topological sorting of the dependency graph.
    _processors: collections.OrderedDict[
        Type[definitions.ABCProcessor], definitions.ProcessorAPIDescriptor
    ]
    _name_to_cls: MutableMapping[str, Type]

    def __init__(self) -> None:
        self._processors = collections.OrderedDict()
        self._name_to_cls = {}

    def register_processor(
        self, processor_descriptor: definitions.ProcessorAPIDescriptor
    ):
        for dep in processor_descriptor.depdendencies.values():
            # To depend on a processor, the class must be defined (module initialized)
            # which entails that is has already been added to the registry.
            assert dep in self._processors, dep

        # Because class are globally unique, to prevent re-use / overwriting of names,
        # We must check this in addition.
        if processor_descriptor.cls_name in self._name_to_cls:
            conflict = self._name_to_cls[processor_descriptor.cls_name]
            existing_source_path = self._processors[conflict].src_path
            raise definitions.APIDefinitonError(
                f"A processor with name `{processor_descriptor.cls_name}` was already "
                f"defined, processors names must be uniuqe. The pre-existing name "
                f"comes from:\n`{existing_source_path}`\nNew conflicting from\n "
                f"{processor_descriptor.src_path}"
            )

        self._processors[processor_descriptor.processor_cls] = processor_descriptor
        self._name_to_cls[
            processor_descriptor.cls_name
        ] = processor_descriptor.processor_cls

    @property
    def processor_descriptors(self) -> list[definitions.ProcessorAPIDescriptor]:
        return list(self._processors.values())

    def get_descriptor(
        self, processor_cls: Type[definitions.ABCProcessor]
    ) -> definitions.ProcessorAPIDescriptor:
        return self._processors[processor_cls]

    def get_dependencies(
        self, processor: definitions.ProcessorAPIDescriptor
    ) -> Iterable[definitions.ProcessorAPIDescriptor]:
        return [
            self._processors[desc]
            for desc in self._processors[processor.processor_cls].depdendencies.values()
        ]


_global_processor_registry = _ProcessorRegistry()


# Processor class runtime utils ########################################################


def _determine_arguments(func: Callable, **kwargs):
    """Merges proivded and default arguments to effective invocation arguments."""
    sig = inspect.signature(func)
    bound_args = sig.bind_partial(**kwargs)
    bound_args.apply_defaults()
    return bound_args.arguments


def ensure_args_are_injected(cls, original_init: Callable, kwargs) -> None:
    """Asserts all placeholder markers are replaced by actual objects."""
    final_args = _determine_arguments(original_init, **kwargs)
    for name, value in final_args.items():
        if isinstance(value, _BaseProvisionPlaceholder):
            raise definitions.UsageError(
                f"When initializing class `{cls.__name__}`, for "
                f"default argument `{name}` a symbolic placeholder value "
                f"was passed (`{value}`). Processors must be either a) locally "
                f"instantiated in `{run_local.__name__}` context or b) deployed "
                "remotely. Naive instantiations are prohibited."
            )


# Local Deployment #####################################################################


def _create_local_context(
    processor_cls: Type[definitions.ABCProcessor],
) -> definitions.Context:
    if hasattr(processor_cls, "default_config"):
        defaults = processor_cls.default_config
        return definitions.Context(user_config=defaults.user_config)
    return definitions.Context()


def _create_modified_init_for_local(
    processor_descriptor: definitions.ProcessorAPIDescriptor,
    cls_to_instance: MutableMapping[
        Type[definitions.ABCProcessor], definitions.ABCProcessor
    ],
):
    """Replaces the default argument values with local processor instantiations.

    If this patch is used, processors can be functionally instantiated without
    any init args (because the patched defaults are sufficient).
    """
    original_init = processor_descriptor.processor_cls.__init__

    def init_for_local(self: definitions.ABCProcessor, **kwargs) -> None:
        logging.debug(f"Patched `__init__` of `{processor_descriptor.cls_name}`.")
        kwargs_mod = dict(kwargs)
        if definitions.CONTEXT_ARG_NAME not in kwargs_mod:
            context = _create_local_context(processor_descriptor.processor_cls)
            kwargs_mod[definitions.CONTEXT_ARG_NAME] = context
        else:
            logging.debug(
                f"Use explicitly given context for `{self.__class__.__name__}`."
            )
        for arg_name, dep_cls in processor_descriptor.depdendencies.items():
            if arg_name in kwargs_mod:
                logging.debug(
                    f"Use explicitly given instance for `{arg_name}` of "
                    f"type `{dep_cls.__name__}`."
                )
                continue
            if dep_cls in cls_to_instance:
                logging.debug(
                    f"Use previously created instace for `{arg_name}` of type "
                    f"`{dep_cls.__name__}`."
                )
                instance = cls_to_instance[dep_cls]
            else:
                logging.debug(
                    f"Create new instace for `{arg_name}` of type `{dep_cls.__name__}`."
                )
                assert dep_cls._init_is_patched
                instance = dep_cls()  # type: ignore  # Here init args are patched.
                cls_to_instance[dep_cls] = instance

            kwargs_mod[arg_name] = instance

        original_init(self, **kwargs_mod)

    return init_for_local


@contextlib.contextmanager
def run_local() -> Any:
    """Context to run processors with depenedency injection from local instances."""
    type_to_instance: MutableMapping[
        Type[definitions.ABCProcessor], definitions.ABCProcessor
    ] = {}
    original_inits: MutableMapping[Type[definitions.ABCProcessor], Callable] = {}

    for processor_descriptor in _global_processor_registry.processor_descriptors:
        original_inits[
            processor_descriptor.processor_cls
        ] = processor_descriptor.processor_cls.__init__
        init_for_local = _create_modified_init_for_local(
            processor_descriptor, type_to_instance
        )
        processor_descriptor.processor_cls.__init__ = init_for_local  # type: ignore[method-assign]
        processor_descriptor.processor_cls._init_is_patched = True
    try:
        yield
    finally:
        # Restore original classes to unpatched state.
        for processor_cls, original_init in original_inits.items():
            processor_cls.__init__ = original_init  # type: ignore[method-assign]
            processor_cls._init_is_patched = False


# Remote Deployment ####################################################################


def _create_remote_service(
    baseten_client: deploy.BasetenClient,
    processor_dir: pathlib.Path,
    workflow_root: pathlib.Path,
    processor_descriptor: definitions.ProcessorAPIDescriptor,
    stub_cls_to_url: Mapping[str, str],
    maybe_stub_file: Optional[pathlib.Path],
    worfklow_name: str,
    generate_only: bool,
) -> definitions.BasetenRemoteDescriptor:
    processor_filepath = shutil.copy(
        processor_descriptor.src_path,
        os.path.join(processor_dir, f"{definitions.PROCESSOR_MODULE}.py"),
    )
    code_gen.generate_processor_source(
        pathlib.Path(processor_filepath), processor_descriptor
    )
    # Only add needed stub URLs.
    stub_cls_to_url = {
        stub_cls.__name__: stub_cls_to_url[stub_cls.__name__]
        for stub_cls in processor_descriptor.depdendencies.values()
    }
    # Convert to truss and deploy.
    # TODO: support file-based config (and/or merge file and python-src configvalues).
    slay_config = processor_descriptor.processor_cls.default_config
    processor_name = slay_config.name or processor_descriptor.cls_name
    model_name = f"{worfklow_name}.{processor_name}"
    truss_dir = deploy.make_truss(
        processor_dir,
        workflow_root,
        slay_config,
        model_name,
        stub_cls_to_url,
        maybe_stub_file,
    )
    if generate_only:
        remote_descriptor = definitions.BasetenRemoteDescriptor(
            b10_model_id="dummy",
            b10_model_name=model_name,
            b10_model_version_id="dymmy",
            b10_model_url="https://dummy",
        )
    else:
        with utils.log_level(logging.INFO):
            remote_descriptor = baseten_client.deploy_truss(truss_dir)
    logging.debug(remote_descriptor)
    return remote_descriptor


def _get_ordered_processor_descriptors(
    processors: Iterable[Type[definitions.ABCProcessor]],
) -> Iterable[definitions.ProcessorAPIDescriptor]:
    """Gather all processors needed and returns a topologically ordered list."""
    needed_processors: set[definitions.ProcessorAPIDescriptor] = set()

    def add_needed_procssors(proc: definitions.ProcessorAPIDescriptor):
        needed_processors.add(proc)
        for processor_descriptor in _global_processor_registry.get_dependencies(proc):
            needed_processors.add(processor_descriptor)
            add_needed_procssors(processor_descriptor)

    for proc_cls in processors:
        proc = _global_processor_registry.get_descriptor(proc_cls)
        add_needed_procssors(proc)

    # Iterating over the registry ensures topological ordering.
    return [
        processor_descriptor
        for processor_descriptor in _global_processor_registry.processor_descriptors
        if processor_descriptor in needed_processors
    ]


def deploy_remotely(
    entrypoint: Type[definitions.ABCProcessor],
    worfklow_name: str,
    baseten_url: str = "https://app.baseten.co",
    generate_only: bool = False,
) -> definitions.BasetenRemoteDescriptor:
    """
    * Gathers dependencies of `entrypoint.
    * Generates stubs.
    * Generates modifies processors to use these stubs.
    * Generates truss models and deploys them to baseten.
    """
    # TODO: more control e.g. publish vs. draft.
    workflow_root = pathlib.Path(sys.argv[0]).absolute().parent
    api_key = deploy.get_api_key_from_truss_config()
    baseten_client = deploy.BasetenClient(baseten_url, api_key)
    entrypoint_descr = _global_processor_registry.get_descriptor(entrypoint)

    ordered_descriptors = _get_ordered_processor_descriptors([entrypoint])
    stub_cls_to_url: dict[str, str] = {}
    entrypoint_remote: Optional[definitions.BasetenRemoteDescriptor] = None
    for processor_descriptor in ordered_descriptors:
        processor_dir = code_gen.make_processor_dir(
            workflow_root, worfklow_name, processor_descriptor
        )
        maybe_stub_file = code_gen.generate_stubs_for_deps(
            processor_dir,
            _global_processor_registry.get_dependencies(processor_descriptor),
        )
        remote_descriptor = _create_remote_service(
            baseten_client,
            processor_dir,
            workflow_root,
            processor_descriptor,
            stub_cls_to_url,
            maybe_stub_file,
            worfklow_name,
            generate_only,
        )
        stub_cls_to_url[processor_descriptor.cls_name] = remote_descriptor.b10_model_url
        if processor_descriptor == entrypoint_descr:
            entrypoint_remote = remote_descriptor

    assert entrypoint_remote is not None
    return entrypoint_remote
