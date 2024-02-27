"""
Open questions - safe translation from processors.py to this generated file.

* How to deal with locally defined symbols from `processors.py`?
    - Force all processors in the same file to use the same image/deps and
      add any non `processor`-definition source and imports as common "header" here.
    - Specifically backtrack which symbols are used (recursively) and only add those.
* Locally defined symbols must not reference processors!
* Can processors reference processors?
    - What about class-processors that are not instantiated in `processors.py`?

"""

import contextlib
import inspect
import logging
from typing import (
    Any,
    Callable,
    ClassVar,
    Generic,
    Iterable,
    Mapping,
    MutableMapping,
    Optional,
    Type,
    TypeVar,
    get_origin,
    get_type_hints,
)

import pydantic

UserConfigT = TypeVar("UserConfigT", pydantic.BaseModel, None)

CONFIG_ARG_NAME = "config"


class APIDefinitonError(TypeError):
    ...


class MissingDependencyError(TypeError):
    ...


class CyclicDependencyError(TypeError):
    ...


class UsageError(RuntimeError):
    ...


class EndpointAPIDescriptor(pydantic.BaseModel):
    name: str
    input_type: Type[pydantic.BaseModel]
    output_type: Type[pydantic.BaseModel]


class ProcessorAPIDescriptor(pydantic.BaseModel):
    # TODO: should class-types, names or full names be used to identify processors?
    # name: str
    processor_cls: Type["BaseProcessor"]
    depdendencies: Mapping[str, Type["BaseProcessor"]]
    endpoints: list[EndpointAPIDescriptor]


def is_function_or_method(obj):
    if inspect.isfunction(obj) or inspect.ismethod(obj):
        return True
    elif hasattr(obj, "__func__") and inspect.isfunction(obj.__func__):
        return True
    return False


def validate_method_names_and_get_endpoint_names(cls) -> set[str]:
    # TODO: differentiate between static, class and instance methods better.
    base_callables: set[str] = set()
    for base in cls.__bases__ + (object,):
        base_callables.update(
            name for name, _ in inspect.getmembers(base, is_function_or_method)
        )

    cls_callables = {name for name, _ in inspect.getmembers(cls, is_function_or_method)}

    added_public_callables = {
        name for name in cls_callables - base_callables if not name.startswith("_")
    }
    if not added_public_callables:
        raise APIDefinitonError(
            f"`{cls.__name__}` must implement at least one additional public method."
        )
    return added_public_callables


def validate_base_classes(
    cls: "Type[BaseProcessor]", endpoint_names: Iterable[str]
) -> None:
    # What if multiple public methods are added?
    ...


def validate_and_describe_endpoint(
    cls: "Type[BaseProcessor]", endpoint_name: str
) -> EndpointAPIDescriptor:
    endpoint_method = getattr(cls, endpoint_name)  # This is the unbound method.
    signature = inspect.signature(endpoint_method)
    params = list(signature.parameters.values())
    assert params[0].name == "self"
    if len(params) > 2:
        raise APIDefinitonError()

    type_hints = get_type_hints(endpoint_method)
    if "return" not in type_hints or params[1].annotation not in type_hints.values():
        raise APIDefinitonError(
            f"Endpoint method `{endpoint_name}` must have a return type annotation."
        )

    input_type = params[1].annotation
    if not issubclass(input_type, pydantic.BaseModel):
        raise APIDefinitonError(
            f"Endpoint method `{endpoint_name}` must have a pydantic model as input data type."
        )
    output_type = type_hints["return"]
    if not issubclass(output_type, pydantic.BaseModel):
        raise APIDefinitonError(
            f"Endpoint method `{endpoint_name}` must have a pydantic model as output data type."
        )

    return EndpointAPIDescriptor(
        name=endpoint_name, input_type=input_type, output_type=output_type
    )


def get_class_type(var):
    origin = get_origin(var)
    return origin if origin is not None else var


def validate_init_signature_and_get_dependencies(
    cls: "Type[BaseProcessor]",
) -> Mapping[str, Type["BaseProcessor"]]:
    signature = inspect.signature(cls.__init__)
    params = list(signature.parameters.values())
    assert params[0].name == "self"
    if len(params) <= 1:
        raise APIDefinitonError()
    if params[1].name != CONFIG_ARG_NAME:
        raise APIDefinitonError(params)
    param_1_type = get_class_type(params[1].annotation)

    if not issubclass(param_1_type, Config):
        raise APIDefinitonError(params)

    depdendencies = {}
    for param in params[2:]:
        # TODO: deal with subclasses, unions, optionals, check default value etc.
        if not issubclass(param.annotation, BaseProcessor):
            raise APIDefinitonError(param)
        if param.annotation in depdendencies:
            raise APIDefinitonError(param)
        depdendencies[param.name] = param.annotation
    return depdendencies


def validate_variable_access(cls: "Type[BaseProcessor]") -> None:
    # Access processors only via `provided` in `__init__`. No globals.
    ...


class _BaseProvisionPlaceholder:
    ...


class ProcessorProvisionPlaceholder(_BaseProvisionPlaceholder):
    def __init__(self, processor_cls: Type["BaseProcessor"]) -> None:
        self._processor_cls = processor_cls


class ConfigProvisionPlaceholder(_BaseProvisionPlaceholder):
    ...


class ProcessorRegistry:
    _processors: MutableMapping[Type["BaseProcessor"], ProcessorAPIDescriptor]

    def __init__(self) -> None:
        self._processors = {}

    def register_processor(self, processor_descriptor: ProcessorAPIDescriptor):
        for dep in processor_descriptor.depdendencies.values():
            if dep not in self._processors:
                raise MissingDependencyError(dep)

        self._processors[processor_descriptor.processor_cls] = processor_descriptor

    @property
    def processor_classes(self) -> list[ProcessorAPIDescriptor]:
        return list(self._processors.values())


_global_processor_registry = ProcessorRegistry()


# Public API ###########################################################################


class Image(pydantic.BaseModel):
    ...


class Config(pydantic.BaseModel, Generic[UserConfigT]):
    name: Optional[str] = None
    image: Optional[Image] = None
    user_config: Optional[UserConfigT] = None


def provide_config() -> Any:
    return ConfigProvisionPlaceholder()


def provide(processor_cls: Type["BaseProcessor"]) -> Any:
    return ProcessorProvisionPlaceholder(processor_cls)


def determine_arguments(func, **kwargs):
    sig = inspect.signature(func)
    bound_args = sig.bind_partial(**kwargs)
    bound_args.apply_defaults()
    return bound_args.arguments


class BaseProcessor(Generic[UserConfigT]):

    default_config: ClassVar[Config]
    _init_is_patched: ClassVar[bool] = False
    _config: Config[UserConfigT]

    def __init__(self, config: Config[UserConfigT] = provide_config()) -> None:
        self._config = config

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        endpoint_names = validate_method_names_and_get_endpoint_names(cls)
        validate_base_classes(cls, endpoint_names)
        endpoint_descriptors = []
        for name in endpoint_names:
            endpoint_descriptors.append(validate_and_describe_endpoint(cls, name))
        dependencies = validate_init_signature_and_get_dependencies(cls)
        processor_descriptor = ProcessorAPIDescriptor(
            processor_cls=cls,
            depdendencies=dependencies,
            endpoints=endpoint_descriptors,
        )
        _global_processor_registry.register_processor(processor_descriptor)

        original_init = cls.__init__

        def modified_init(self, *args, **kwargs):
            final_args = determine_arguments(original_init, **kwargs)
            for name, value in final_args.items():
                if isinstance(value, _BaseProvisionPlaceholder):
                    raise UsageError(
                        f"When initializing class `{cls.__name__}`, for "
                        f"default argument `{name}` a symbolic placeholder value "
                        "was passed. It is required to instantiate and run the classes "
                        f"in a `{run_local.__name__}` context or deploy them remotely."
                    )

            original_init(self, *args, **kwargs)

        cls.__init__ = modified_init  # type: ignore[method-assign]


# Local Deployment #####################################################################


def _create_modified_init(
    processor_descriptor: ProcessorAPIDescriptor,
    cls_to_instance: MutableMapping[Type[BaseProcessor], BaseProcessor],
):
    original_init = processor_descriptor.processor_cls.__init__

    def modified_init(self: BaseProcessor, **kwargs) -> None:
        print(f"Patched `__init__` of `{processor_descriptor.processor_cls}`.")
        if hasattr(processor_descriptor.processor_cls, "default_config"):
            config = processor_descriptor.processor_cls.default_config
        else:
            config = Config()

        for arg_name, dep_cls in processor_descriptor.depdendencies.items():
            if arg_name in kwargs:
                logging.debug(
                    f"Use explicitly given instance for `{arg_name}` of "
                    f"type `{dep_cls.__name__}`."
                )
                continue

            if dep_cls in cls_to_instance:
                logging.debug(
                    f"Use mapped instace for `{arg_name}` of type `{dep_cls.__name__}`."
                )
                instance = cls_to_instance[dep_cls]
            else:
                logging.debug(
                    f"Create new instace for `{arg_name}` of type `{dep_cls.__name__}`."
                )
                assert dep_cls._init_is_patched
                instance = dep_cls()
                cls_to_instance[dep_cls] = instance

            kwargs[arg_name] = instance

        original_init(self, config=config, **kwargs)

    return modified_init


@contextlib.contextmanager
def run_local() -> Any:
    type_to_instance: MutableMapping[Type[BaseProcessor], BaseProcessor] = {}
    original_inits: MutableMapping[Type[BaseProcessor], Callable] = {}

    for processor_descriptor in _global_processor_registry.processor_classes:
        original_inits[
            processor_descriptor.processor_cls
        ] = processor_descriptor.processor_cls.__init__
        patched_init = _create_modified_init(processor_descriptor, type_to_instance)
        processor_descriptor.processor_cls.__init__ = patched_init  # type: ignore[method-assign]
        processor_descriptor.processor_cls._init_is_patched = True
    try:
        yield
    finally:
        # Restore original classes to unpatched state.
        for processor_cls, original_init in original_inits.items():
            processor_cls.__init__ = original_init  # type: ignore[method-assign]
            processor_cls._init_is_patched = False


def deploy_remotely(processors: Iterable[Type[BaseProcessor]]) -> None:
    ...
