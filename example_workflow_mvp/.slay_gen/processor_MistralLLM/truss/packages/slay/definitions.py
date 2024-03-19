import abc
from types import GenericAlias
from typing import Any, ClassVar, Generic, Mapping, Optional, Type, TypeVar

import pydantic
from truss.templates.shared import secrets_resolver

UserConfigT = TypeVar("UserConfigT", bound=Optional[pydantic.BaseModel])

BASTEN_APY_SECRET_NAME = "baseten_api_key"
ENDPOINT_NAME = "run"  # Referring to processors method name exposed as endpoint.


class APIDefinitonError(TypeError):
    """Raised when user-defined processors do not adhere to API constraints."""


class MissingDependencyError(TypeError):
    """Raised when a needed resource could not be found or is not defined."""


class UsageError(Exception):
    """Raised when components are not used the expected way at runtime."""


class Image(pydantic.BaseModel):
    # TODO: this is a placeholder/dummy object.

    def pip_requirements_txt(self, *args, **kwargs) -> "Image":
        return self

    def pip_install(self, *args, **kwargs) -> "Image":
        return self

    def cuda(self, *args, **kwargs) -> "Image":
        return self


class Resources(pydantic.BaseModel):
    # TODO: this is a placeholder/dummy object.

    def cpu(self, *args, **kwargs) -> "Resources":
        return self

    def gpu(self, *args, **kwargs) -> "Resources":
        return self


class Config(pydantic.BaseModel, Generic[UserConfigT]):
    """Bundles config values needed to deploy a processor."""

    name: Optional[str] = None
    image: Optional[Image] = None
    resources: Optional[Resources] = None
    user_config: UserConfigT = pydantic.Field(default=None)


class Context(pydantic.BaseModel, Generic[UserConfigT]):
    """Bundles config values needed to instantiate a processor in deployment."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    user_config: UserConfigT = pydantic.Field(default=None)
    stub_cls_to_url: dict[str, str] = {}
    # secrets: Optional[secrets_resolver.Secrets] = None
    # TODO: above type results in `truss.server.shared.secrets_resolver.Secrets`
    # due to the templating, at runtime the object passed will be from
    # `shared.secrets_resolver` and give pydantic validation error.
    secrets: Optional[Any] = None

    def get_stub_url(self, stub_cls_name: str) -> str:
        if stub_cls_name not in self.stub_cls_to_url:
            raise MissingDependencyError(f"{stub_cls_name}")
        return self.stub_cls_to_url[stub_cls_name]

    def get_baseten_api_key(self) -> str:
        if not self.secrets:
            raise UsageError(f"Secrets not set in `{self.__class__.__name__}` object.")
        if BASTEN_APY_SECRET_NAME not in self.secrets:
            raise MissingDependencyError(
                "For using workflows, it is required to setup a an API key with name "
                f"`{BASTEN_APY_SECRET_NAME}` on baseten to allow workflow processor to "
                "call other processors."
            )

        api_key = self.secrets[BASTEN_APY_SECRET_NAME]
        return api_key


class TrussMetadata(pydantic.BaseModel, Generic[UserConfigT]):
    """Plugin for the truss config (in config["model_metadata"]["slay_metadata"])."""

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

    # Cannot add this abstract method to API, because overriding it with concrete
    # arguments would give type error.s
    # @abc.abstractmethod
    # def predict(self, *args, **kwargs) -> Any: ...

    @property
    @abc.abstractmethod
    def user_config(self) -> UserConfigT:
        ...


class TypeDescriptor(pydantic.BaseModel):
    """For describing I/O types of processors."""

    raw: Any  # The raw type annotation object (could be a type or GenericAlias).

    def as_src_str(self) -> str:
        if isinstance(self.raw, type):
            return self.raw.__name__
        else:
            return str(self.raw)

    @property
    def is_pydantic(self) -> bool:
        return (
            isinstance(self.raw, type)
            and not isinstance(self.raw, GenericAlias)
            and issubclass(self.raw, pydantic.BaseModel)
        )


class EndpointAPIDescriptor(pydantic.BaseModel):
    name: str = ENDPOINT_NAME
    input_names_and_tyes: list[tuple[str, TypeDescriptor]]
    output_types: list[TypeDescriptor]
    is_async: bool
    is_generator: bool


class ProcessorAPIDescriptor(pydantic.BaseModel):
    processor_cls: Type[ABCProcessor]
    src_path: str
    depdendencies: Mapping[str, Type[ABCProcessor]]
    endpoint: EndpointAPIDescriptor

    def __hash__(self) -> int:
        return hash(self.processor_cls)

    @property
    def cls_name(self) -> str:
        return self.processor_cls.__name__


class BasetenRemoteDescriptor(pydantic.BaseModel):
    b10_model_id: str
    b10_model_version_id: str
    b10_model_name: str
    b10_model_url: str
