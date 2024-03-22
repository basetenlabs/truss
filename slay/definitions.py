import abc
from types import GenericAlias
from typing import Any, ClassVar, Generic, Mapping, Optional, Type, TypeVar

import pydantic
from pydantic import generics

UserConfigT = TypeVar("UserConfigT", bound=Optional[pydantic.BaseModel])

BASTEN_APY_SECRET_NAME = "baseten_api_key"
TRUSS_CONFIG_SLAY_KEY = "slay_metadata"

ENDPOINT_NAME = "run"  # Referring to processor method name exposed as endpoint.
# Below arg names must correspond to `definitions.ABCProcessor`.
CONTEXT_ARG_NAME = "context"  # Referring to processors `__init__` signature.
SELF_ARG_NAME = "self"

GENERATED_CODE_DIR = ".slay_generated"


class APIDefinitonError(TypeError):
    """Raised when user-defined processors do not adhere to API constraints."""


class MissingDependencyError(TypeError):
    """Raised when a needed resource could not be found or is not defined."""


class UsageError(Exception):
    """Raised when components are not used the expected way at runtime."""


class ImageSpec(pydantic.BaseModel):
    base_image: str = "python:3.11-slim"
    pip_requirements_file: Optional[str] = None
    pip_requirements: list[str] = []
    apt_requirements: list[str] = []


class Image:
    _spec: ImageSpec

    def __init__(self) -> None:
        self._spec = ImageSpec()

    def pip_requirements_file(self, file_path: str) -> "Image":
        # TODO: deal with relative paths.
        self._spec.pip_requirements_file = file_path
        return self

    def pip_requirements(self, requirements: list[str]) -> "Image":
        self._spec.pip_requirements = requirements
        return self

    def apt_requirements(self, requirements: list[str]) -> "Image":
        self._spec.apt_requirements = requirements
        return self

    def get_spec(self) -> ImageSpec:
        return self._spec.copy(deep=True)


class ComputeSpec(pydantic.BaseModel):
    cpu: str = "1"
    memory: str = "2Gi"
    gpu: Optional[str] = None


class Compute:
    _spec: ComputeSpec

    def __init__(self) -> None:
        self._spec = ComputeSpec()

    def cpu(self, cpu: int) -> "Compute":
        self._spec.cpu = str(cpu)
        return self

    def memory(self, memory: str) -> "Compute":
        self._spec.memory = memory
        return self

    def gpu(self, kind: str, count: int = 1) -> "Compute":
        self._spec.gpu = f"{kind}:{count}"
        return self

    def get_spec(self) -> ComputeSpec:
        return self._spec.copy(deep=True)


class AssetSpec(pydantic.BaseModel):
    secrets: dict[str, str] = {}
    cached: list[Any] = []


class Assets:
    _spec: AssetSpec

    def __init__(self) -> None:
        self._spec = AssetSpec()

    def secret(self, key: str) -> "Assets":
        self._spec.secrets[key] = "***"
        return self

    def cached(self, value: list[Any]) -> "Assets":
        self._spec.cached = value
        return self

    def get_spec(self) -> AssetSpec:
        return self._spec.copy(deep=True)


class Config(generics.GenericModel, Generic[UserConfigT]):
    """Bundles config values needed to deploy a processor."""

    class Config:
        arbitrary_types_allowed = True

    name: Optional[str] = None
    image: Image = Image()
    compute: Compute = Compute()
    assets: Assets = Assets()
    user_config: UserConfigT = pydantic.Field(default=None)

    def get_image_spec(self) -> ImageSpec:
        return self.image.get_spec()

    def get_compute_spec(self) -> ComputeSpec:
        return self.compute.get_spec()

    def get_asset_spec(self) -> AssetSpec:
        return self.assets.get_spec()


class Context(generics.GenericModel, Generic[UserConfigT]):
    """Bundles config values needed to instantiate a processor in deployment."""

    class Config:
        arbitrary_types_allowed = True

    user_config: UserConfigT = pydantic.Field(default=None)
    stub_cls_to_url: Mapping[str, str] = {}
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


class TrussMetadata(generics.GenericModel, Generic[UserConfigT]):
    """Plugin for the truss config (in config["model_metadata"]["slay_metadata"])."""

    class Config:
        arbitrary_types_allowed = True

    user_config: UserConfigT = pydantic.Field(default=None)
    stub_cls_to_url: Mapping[str, str] = {}


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
    user_config_type: TypeDescriptor

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


PREDICT_ENDPOINT = "/predict"
PROCESSOR_MODULE = "processor"
