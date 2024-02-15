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

import inspect
from typing import (
    Any,
    ClassVar,
    Generic,
    Iterable,
    Optional,
    Type,
    TypeVar,
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


class EndpointAPIDescriptor(pydantic.BaseModel):
    name: str
    input_type: Type[pydantic.BaseModel]
    output_type: Type[pydantic.BaseModel]


class ProcessorAPIDescriptor(pydantic.BaseModel):
    # TODO: should class-types, names or full names be used to identify processors?
    # name: str
    processor_cls: Type["BaseProcessor"]
    depdendencies: list[Type["BaseProcessor"]]
    endpoints: list[EndpointAPIDescriptor]


def is_function_or_method(obj):
    if inspect.isfunction(obj) or inspect.ismethod(obj):
        return True
    elif hasattr(obj, "__func__") and inspect.isfunction(obj.__func__):
        return True
    return False


def validate_method_names_and_get_endpoint_names(cls) -> set[str]:
    # TODO: differentiate between static, class and instance methods better.
    base_callables = set()
    for base in cls.__bases__ + (object):
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


def validate_init_signature_and_get_dependencies(
    cls: "Type[BaseProcessor]",
) -> list[Type["BaseProcessor"]]:
    signature = inspect.signature(cls.__init__)
    params = list(signature.parameters.values())
    assert params[0].name == "self"
    if len(params) <= 1:
        raise APIDefinitonError()
    if params[1].name != CONFIG_ARG_NAME:
        raise APIDefinitonError(params)
    if not issubclass(params[1].annotation, Config):
        raise APIDefinitonError(params)

    depdendencies = set()
    for param in params[2:]:
        # TODO: deal with subclasses, unions, optionals, check default value etc.
        if not issubclass(param.annotation, BaseProcessor):
            raise APIDefinitonError(param)
        if param.annotation in depdendencies:
            raise APIDefinitonError(param)
        depdendencies.add(param.annotation)
    return list(depdendencies)


def validate_variable_access(cls: "Type[BaseProcessor]") -> None:
    # Access processors only via provion in `__init__`. No globals.
    ...


class ProvisionPlaceholder:
    def __init__(self, processor_cls: Type["BaseProcessor"]) -> None:
        self._processor_cls = processor_cls


class ProcessorRegistry:
    def __init__(self) -> None:
        self._processors = {}

    def register_processor(self, processor_descriptor: ProcessorAPIDescriptor):
        for dep in processor_descriptor.depdendencies:
            if dep not in self._processors:
                raise MissingDependencyError(dep)

        self._processors[processor_descriptor.processor_cls] = processor_descriptor


_global_processor_registry = ProcessorRegistry()

# Public API ###########################################################################


class Image(pydantic.BaseModel):
    ...


class Config(pydantic.BaseModel, Generic[UserConfigT]):
    name: Optional[str] = None
    image: Optional[Image] = None
    user_config: Optional[UserConfigT] = None


class BaseProcessor(Generic[UserConfigT]):

    default_config: ClassVar[Config]
    _config: Config[UserConfigT]

    def __init__(self, config: Config[UserConfigT]) -> None:
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


def provide(processor_cls: Type[BaseProcessor]) -> Any:
    return ProvisionPlaceholder(processor_cls)


def run_local() -> Any:
    ...


def deploy_remotely(processors: Iterable[Type[BaseProcessor]]) -> None:
    ...
