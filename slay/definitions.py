import abc
from typing import Any, ClassVar, Generic, Mapping, Optional, Type, TypeVar

import pydantic

UserConfigT = TypeVar("UserConfigT", bound=Optional[pydantic.BaseModel])


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

    def pip_requirements_txt(self, *args, **kwargs) -> "Image":
        return self

    def pip_install(self, *args, **kwargs) -> "Image":
        return self

    def cuda(self, *args, **kwargs) -> "Image":
        return self


class Resources(pydantic.BaseModel):
    ...

    def cpu(self, *args, **kwargs) -> "Resources":
        return self

    def gpu(self, *args, **kwargs) -> "Resources":
        return self


class Config(pydantic.BaseModel, Generic[UserConfigT]):
    name: Optional[str] = None
    image: Optional[Image] = None
    resources: Optional[Resources] = None
    user_config: UserConfigT = pydantic.Field(default=None)


class ABCProcessor(Generic[UserConfigT], abc.ABC):
    default_config: ClassVar[Config]
    _init_is_patched: ClassVar[bool] = False
    _config: Config[UserConfigT]

    @abc.abstractmethod
    def __init__(self, config: Config[UserConfigT]) -> None:
        ...

    @property
    @abc.abstractmethod
    def user_config(self) -> UserConfigT:
        ...


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
