import abc
from typing import Any, ClassVar, Generic, Mapping, Optional, Type, TypeVar

import pydantic

UserConfigT = TypeVar("UserConfigT", pydantic.BaseModel, None)


class APIDefinitonError(TypeError):
    ...


class MissingDependencyError(TypeError):
    ...


class CyclicDependencyError(TypeError):
    ...


class UsageError(Exception):
    ...


class Image(pydantic.BaseModel):
    ...


class Config(pydantic.BaseModel, Generic[UserConfigT]):
    name: Optional[str] = None
    image: Optional[Image] = None
    user_config: Optional[UserConfigT] = None


class ABCProcessor(abc.ABC, Generic[UserConfigT]):
    default_config: ClassVar[Config]
    _init_is_patched: ClassVar[bool] = False
    _config: Config[UserConfigT]

    def __init__(self, config: Config[UserConfigT]) -> None:
        self._config = config


class EndpointAPIDescriptor(pydantic.BaseModel):
    name: str
    input_types: list[Type]  # Type[pydantic.BaseModel]
    output_type: Any  # GenericAlias | Type  # Type[pydantic.BaseModel]
    is_async: bool
    is_generator: bool


class ProcessorAPIDescriptor(pydantic.BaseModel):
    processor_cls: Type[ABCProcessor]
    depdendencies: Mapping[str, Type[ABCProcessor]]
    endpoints: list[EndpointAPIDescriptor]
