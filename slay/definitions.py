import abc
from typing import Any, ClassVar, Generic, Mapping, Optional, Type, TypeVar, final

import pydantic
from truss.templates.shared import secrets_resolver

UserConfigT = TypeVar("UserConfigT", bound=Optional[pydantic.BaseModel])

BASTEN_APY_KEY_NAME = "baseten_api_key"


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


class Context(pydantic.BaseModel, Generic[UserConfigT]):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    user_config: UserConfigT = pydantic.Field(default=None)
    stub_cls_to_url: dict[str, str] = {}
    # secrets: Optional[secrets_resolver.Secrets] = None
    # TODO: above type results in `truss.server.shared.secrets_resolver.Secrets`
    # due to the templating, at runtime the object passed will be from
    # `shared.secrets_resolver` and give pydantic validation error.
    secrets: Optional[Any] = None

    def get_stub_url(self, stub_cls: Type) -> str:
        stub_cls_name = stub_cls.__name__
        if stub_cls_name not in self.stub_cls_to_url:
            raise MissingDependencyError(f"{stub_cls_name}")
        return self.stub_cls_to_url[stub_cls_name]

    def get_baseten_api_key(self) -> str:
        if not self.secrets:
            raise ValueError(f"Secrets not provided")
        if BASTEN_APY_KEY_NAME not in self.secrets:
            raise MissingDependencyError(f"{BASTEN_APY_KEY_NAME}")

        maybe_key = self.secrets.get(BASTEN_APY_KEY_NAME)
        if not maybe_key:
            raise MissingDependencyError(f"{BASTEN_APY_KEY_NAME}")
        return maybe_key


class TrussMetadata(pydantic.BaseModel, Generic[UserConfigT]):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    user_config: UserConfigT = pydantic.Field(default=None)
    stub_cls_to_url: dict[str, str] = {}


class ABCProcessor(Generic[UserConfigT], abc.ABC):
    default_config: ClassVar[Config]
    _init_is_patched: ClassVar[bool] = False
    _context: Context[UserConfigT]

    @abc.abstractmethod
    def __init__(self, context: Context[UserConfigT]) -> None:
        ...

    @property
    @abc.abstractmethod
    def user_config(self) -> UserConfigT:
        ...


class TypeDescriptor(pydantic.BaseModel):
    raw: Any

    def as_str(self) -> str:
        if isinstance(self.raw, type):
            return self.raw.__name__
        else:
            return str(self.raw)

    @property
    def is_pydantic(self) -> bool:
        return isinstance(self.raw, type) and issubclass(self.raw, pydantic.BaseModel)


class EndpointAPIDescriptor(pydantic.BaseModel):
    name: str
    input_name_and_tyes: list[tuple[str, TypeDescriptor]]
    output_types: list[TypeDescriptor]
    is_async: bool
    is_generator: bool


class ProcessorAPIDescriptor(pydantic.BaseModel):
    processor_cls: Type[ABCProcessor]
    src_file: str
    depdendencies: Mapping[str, Type[ABCProcessor]]
    endpoints: list[EndpointAPIDescriptor]  # Initially just one.

    def __hash__(self) -> int:
        return hash(self.processor_cls)

    @property
    def cls_name(self) -> str:
        return self.processor_cls.__name__


class BasetenRemoteDescriptor(pydantic.BaseModel):
    b10_model_id: str
    b10_model_version_id: str
    b10_model_name: str
    b10_predict_url: str
