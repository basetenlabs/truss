# TODO: this file contains too much implementation -> restructure.
import abc
import logging
import os
import pathlib
import traceback
from types import GenericAlias
from typing import (
    Any,
    ClassVar,
    Generic,
    Iterable,
    Mapping,
    Optional,
    Type,
    TypeVar,
    Union,
    cast,
)

import pydantic
from pydantic import generics
from truss import truss_config
from truss.remote import baseten as baseten_remote
from truss.remote import remote_cli, remote_factory

UserConfigT = TypeVar("UserConfigT", bound=Optional[pydantic.BaseModel])

BASETEN_API_SECRET_NAME = "baseten_chain_api_key"
SECRET_DUMMY = "***"
TRUSS_CONFIG_CHAINS_KEY = "chains_metadata"

GENERATED_CODE_DIR = ".chains_generated"
PREDICT_ENDPOINT_NAME = "/predict"

# Below arg names must correspond to `definitions.ABCChainlet`.
ENDPOINT_METHOD_NAME = "run"  # Referring to Chainlet method name exposed as endpoint.
CONTEXT_ARG_NAME = "context"  # Referring to Chainlets `__init__` signature.
SELF_ARG_NAME = "self"
REMOTE_CONFIG_NAME = "remote_config"


class SafeModel(pydantic.BaseModel):
    """Pydantic base model with reasonable config."""

    class Config:
        arbitrary_types_allowed = False
        validate_all = True
        validate_assignment = True
        extra = pydantic.Extra.forbid


class SafeModelNonSerializable(pydantic.BaseModel):
    """Pydantic base model with reasonable config - allowing arbitrary types."""

    class Config:
        arbitrary_types_allowed = True
        validate_all = True
        validate_assignment = True
        extra = pydantic.Extra.forbid


class ChainsUsageError(TypeError):
    """Raised when user-defined Chainlets do not adhere to API constraints."""


class MissingDependencyError(TypeError):
    """Raised when a needed resource could not be found or is not defined."""


class ChainsRuntimeError(Exception):
    """Raised when components are not used the expected way at runtime."""


class AbsPath:
    _abs_file_path: str
    _creating_module: str
    _original_path: str

    def __init__(
        self, abs_file_path: str, creating_module: str, original_path: str
    ) -> None:
        self._abs_file_path = abs_file_path
        self._creating_module = creating_module
        self._original_path = original_path

    def _raise_if_not_exists(self, abs_path: str) -> None:
        if not os.path.isfile(abs_path):
            raise MissingDependencyError(
                f"With the file path `{self._original_path}` an absolute path relative "
                f"to the calling module `{self._creating_module}` was created, "
                f"resulting `{self._abs_file_path}` - but no file was found."
            )

    @property
    def abs_path(self) -> str:
        if self._abs_file_path != self._original_path:
            logging.info(
                f"Using abs path `{self._abs_file_path}` for path specified as "
                f"`{self._original_path}` (in `{self._creating_module}`)."
            )
        abs_path = self._abs_file_path
        self._raise_if_not_exists(abs_path)
        return abs_path


class DockerImage(SafeModelNonSerializable):
    # TODO: this is not stable yet and might change or refer back to truss.
    base_image: str = "python:3.11-slim"
    pip_requirements_file: Optional[AbsPath] = None
    pip_requirements: list[str] = []
    apt_requirements: list[str] = []
    data_dir: Optional[AbsPath] = None


class ComputeSpec(pydantic.BaseModel):
    # TODO: this is not stable yet and might change or refer back to truss.
    cpu_count: int = 1
    memory: str = "2Gi"
    accelerator: truss_config.AcceleratorSpec = truss_config.AcceleratorSpec()


class Compute:
    """Builder to create ComputeSpec."""

    # This extra layer around `ComputeSpec` is needed to parse the accelerator options.

    _spec: ComputeSpec

    def __init__(
        self,
        cpu_count: int = 1,
        memory: str = "2Gi",
        gpu: Union[str, truss_config.Accelerator, None] = None,
        gpu_count: int = 1,
    ) -> None:
        accelerator = truss_config.AcceleratorSpec()
        if gpu:
            accelerator.accelerator = truss_config.Accelerator(gpu)
            accelerator.count = gpu_count
            accelerator = truss_config.AcceleratorSpec(
                accelerator=truss_config.Accelerator(gpu), count=gpu_count
            )

        self._spec = ComputeSpec(
            cpu_count=cpu_count, memory=memory, accelerator=accelerator
        )

    def get_spec(self) -> ComputeSpec:
        return self._spec.copy(deep=True)


class AssetSpec(SafeModel):
    # TODO: this is not stable yet and might change or refer back to truss.
    secrets: dict[str, str] = {}
    cached: list[Any] = []


class Assets:
    """Builder to create asset spec."""

    # This extra layer around `ComputeSpec` is needed to add secret_keys.
    _spec: AssetSpec

    def __init__(
        self,
        cached: Iterable[Any] = (),
        secret_keys: Iterable[str] = (),
    ) -> None:
        self._spec = AssetSpec(
            cached=list(cached), secrets={k: SECRET_DUMMY for k in secret_keys}
        )

    def get_spec(self) -> AssetSpec:
        return self._spec.copy(deep=True)


class RemoteConfig(SafeModelNonSerializable):
    """Bundles config values needed to deploy a Chainlet."""

    docker_image: DockerImage
    compute: Compute = Compute()
    assets: Assets = Assets()
    name: Optional[str] = None

    def get_compute_spec(self) -> ComputeSpec:
        return self.compute.get_spec()

    def get_asset_spec(self) -> AssetSpec:
        return self.assets.get_spec()


class RPCOptions(SafeModel):
    timeout_sec: int = 600
    retries: int = 1


class ServiceDescriptor(SafeModel):
    name: str
    predict_url: str
    options: RPCOptions


class DeploymentContext(generics.GenericModel, Generic[UserConfigT]):
    """Bundles config values and resources needed to instantiate Chainlets."""

    class Config:
        arbitrary_types_allowed = True
        validate_all = True
        validate_assignment = True
        extra = pydantic.Extra.forbid

    data_dir: Optional[pathlib.Path] = None
    user_config: UserConfigT = pydantic.Field(default=None)
    chainlet_to_service: Mapping[str, ServiceDescriptor] = {}
    # secrets: Optional[secrets_resolver.Secrets] = None
    # TODO: above type results in `truss.server.shared.secrets_resolver.Secrets`
    #   due to the templating, at runtime the object passed will be from
    #   `shared.secrets_resolver` and give pydantic validation error.
    secrets: Optional[Any] = None

    def get_service_descriptor(self, chainlet_name: str) -> ServiceDescriptor:
        if chainlet_name not in self.chainlet_to_service:
            raise MissingDependencyError(f"{chainlet_name}")
        return self.chainlet_to_service[chainlet_name]

    def get_baseten_api_key(self) -> str:
        if self.secrets is None:
            raise ChainsRuntimeError(
                f"Secrets not set in `{self.__class__.__name__}` object."
            )
        error_msg = (
            "For using chains, it is required to setup a an API key with name "
            f"`{BASETEN_API_SECRET_NAME}` on baseten to allow chain Chainlet to "
            "call other Chainlets. For local execution, secrets can be provided "
            "to `run_local`."
        )
        if BASETEN_API_SECRET_NAME not in self.secrets:
            raise MissingDependencyError(error_msg)

        api_key = self.secrets[BASETEN_API_SECRET_NAME]
        if api_key == SECRET_DUMMY:
            raise MissingDependencyError(
                f"{error_msg}. Retrieved dummy value of `{api_key}`."
            )
        return api_key


class TrussMetadata(generics.GenericModel, Generic[UserConfigT]):
    """Plugin for the truss config (in config["model_metadata"]["chains_metadata"])."""

    user_config: UserConfigT = pydantic.Field(default=None)
    chainlet_to_service: Mapping[str, ServiceDescriptor] = {}


class ABCChainlet(Generic[UserConfigT], abc.ABC):
    remote_config: ClassVar[RemoteConfig]
    default_user_config: ClassVar[Optional[pydantic.BaseModel]] = None
    _init_is_patched: ClassVar[bool] = False

    _context: DeploymentContext[UserConfigT]

    @abc.abstractmethod
    def __init__(self, context: DeploymentContext[UserConfigT]) -> None:
        ...

    # Cannot add this abstract method to API, because we want to allow arbitrary
    # arg/kwarg names and specifying any function signature here would give type errors
    # @abc.abstractmethod
    # def run(self, *args, **kwargs) -> Any: ...

    @property
    @abc.abstractmethod
    def user_config(self) -> UserConfigT:
        ...


class TypeDescriptor(SafeModelNonSerializable):
    """For describing I/O types of Chainlets."""

    raw: Any  # The raw type annotation object (could be a type or GenericAlias).

    @property
    def is_pydantic(self) -> bool:
        return (
            isinstance(self.raw, type)
            and not isinstance(self.raw, GenericAlias)
            and issubclass(self.raw, pydantic.BaseModel)
        )


class EndpointAPIDescriptor(SafeModelNonSerializable):
    name: str = ENDPOINT_METHOD_NAME
    input_names_and_types: list[tuple[str, TypeDescriptor]]
    output_types: list[TypeDescriptor]
    is_async: bool
    is_generator: bool


class DependencyDescriptor(SafeModelNonSerializable):
    chainlet_cls: Type[ABCChainlet]
    options: RPCOptions

    @property
    def name(self) -> str:
        return self.chainlet_cls.__name__


class ChainletAPIDescriptor(SafeModelNonSerializable):
    chainlet_cls: Type[ABCChainlet]
    src_path: str
    dependencies: Mapping[str, DependencyDescriptor]
    endpoint: EndpointAPIDescriptor
    user_config_type: TypeDescriptor

    def __hash__(self) -> int:
        return hash(self.chainlet_cls)

    @property
    def name(self) -> str:
        return self.chainlet_cls.__name__


class StackFrame(SafeModel):
    filename: str
    lineno: Optional[int]
    name: str
    line: Optional[str]

    @classmethod
    def from_frame_summary(cls, frame: traceback.FrameSummary):
        return cls(
            filename=frame.filename,
            lineno=frame.lineno,
            name=frame.name,
            line=frame.line,
        )

    def to_frame_summary(self) -> traceback.FrameSummary:
        return traceback.FrameSummary(
            filename=self.filename, lineno=self.lineno, name=self.name, line=self.line
        )


class RemoteErrorDetail(SafeModel):
    remote_name: str
    exception_cls_name: str
    exception_module_name: Optional[str]
    exception_message: str
    user_stack_trace: list[StackFrame]

    def to_stack_summary(self) -> traceback.StackSummary:
        return traceback.StackSummary.from_list(
            frame.to_frame_summary() for frame in self.user_stack_trace
        )

    def format(self) -> str:
        stack = "".join(traceback.format_list(self.to_stack_summary()))
        exc_info = (
            f"\n(Exception class defined in `{self.exception_module_name}`.)"
            if self.exception_module_name
            else ""
        )
        error = (
            f"{RemoteErrorDetail.__name__} in `{self.remote_name}`\n"
            f"Traceback (most recent call last):\n"
            f"{stack}{self.exception_cls_name}: {self.exception_message}{exc_info}"
        )
        return error


class GenericRemoteException(Exception):
    ...


########################################################################################


class DeploymentOptions(SafeModelNonSerializable):
    chain_name: str
    only_generate_trusses: bool = False


class DeploymentOptionsBaseten(DeploymentOptions):
    remote_provider: baseten_remote.BasetenRemote
    publish: bool
    promote: bool

    @classmethod
    def create(
        cls,
        chain_name: str,
        publish: bool,
        promote: bool,
        only_generate_trusses: bool,
        remote: Optional[str] = None,
    ) -> "DeploymentOptionsBaseten":
        if not remote:
            remote = remote_cli.inquire_remote_name(
                remote_factory.RemoteFactory.get_available_config_names()
            )
        remote_provider = cast(
            baseten_remote.BasetenRemote,
            remote_factory.RemoteFactory.create(remote=remote),
        )
        return DeploymentOptionsBaseten(
            remote_provider=remote_provider,
            chain_name=chain_name,
            publish=publish,
            promote=promote,
            only_generate_trusses=only_generate_trusses,
        )


class DeploymentOptionsLocalDocker(DeploymentOptions):
    # Local docker-to-docker requests don't need auth, but we need to set a
    # value different from `SECRET_DUMMY` to not trigger the check that the secret
    # is unset. Additionally, if local docker containers make calls to models deployed
    # on baseten, a real API key must be provided (i.e. the default must be overridden).
    baseten_chain_api_key: str = "docker_dummy_key"
