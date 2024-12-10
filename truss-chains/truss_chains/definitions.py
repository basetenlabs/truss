# TODO: this file contains too much implementation -> restructure.
import abc
import enum
import logging
import pathlib
import traceback
from typing import (  # type: ignore[attr-defined]  # Chains uses Python >=3.9.
    Any,
    Callable,
    ClassVar,
    Generic,
    GenericAlias,  # This causes above type error.
    Iterable,
    Literal,
    Mapping,
    Optional,
    Protocol,
    Type,
    TypeVar,
    Union,
    cast,
    runtime_checkable,
)

import pydantic
from truss.base import truss_config
from truss.base.constants import PRODUCTION_ENVIRONMENT_NAME
from truss.remote import baseten as baseten_remote
from truss.remote import remote_factory

BASETEN_API_SECRET_NAME = "baseten_chain_api_key"
SECRET_DUMMY = "***"
TRUSS_CONFIG_CHAINS_KEY = "chains_metadata"
GENERATED_CODE_DIR = ".chains_generated"
DYNAMIC_CHAINLET_CONFIG_KEY = "dynamic_chainlet_config"
OTEL_TRACE_PARENT_HEADER_KEY = "traceparent"
# Below arg names must correspond to `definitions.ABCChainlet`.
ENDPOINT_METHOD_NAME = "run_remote"  # Chainlet method name exposed as endpoint.
CONTEXT_ARG_NAME = "context"  # Referring to Chainlets `__init__` signature.
SELF_ARG_NAME = "self"
REMOTE_CONFIG_NAME = "remote_config"

K = TypeVar("K", contravariant=True)
V = TypeVar("V", covariant=True)


C = TypeVar("C")


class _classproperty(Generic[C, V]):
    def __init__(self, fget: Callable[[Type[C]], V]) -> None:
        self._fget = fget

    def __get__(self, instance: object, owner: Type[C]) -> V:
        return self._fget.__get__(None, owner)()


def classproperty(fget: Callable[[Type[C]], V]) -> _classproperty[C, V]:
    return _classproperty(fget)


@runtime_checkable
class MappingNoIter(Protocol[K, V]):
    def __getitem__(self, key: K) -> V: ...

    def __len__(self) -> int: ...

    def __contains__(self, key: K) -> bool: ...


class SafeModel(pydantic.BaseModel):
    """Pydantic base model with reasonable config."""

    model_config = pydantic.ConfigDict(
        arbitrary_types_allowed=False,
        strict=True,
        validate_assignment=True,
    )


class SafeModelNonSerializable(pydantic.BaseModel):
    """Pydantic base model with reasonable config - allowing arbitrary types."""

    model_config = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
        strict=True,
        validate_assignment=True,
        extra="forbid",
    )


class ChainsUsageError(TypeError):
    """Raised when user-defined Chainlets do not adhere to API constraints."""


class MissingDependencyError(TypeError):
    """Raised when a needed resource could not be found or is not defined."""


class ChainsRuntimeError(Exception):
    """Raised when components are not used the expected way at runtime."""


class ChainsDeploymentError(Exception):
    """Raised when interaction with a Chain deployment are not possible."""


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
        path = pathlib.Path(abs_path)
        if not (path.is_file() or (path.is_dir() and any(path.iterdir()))):
            raise MissingDependencyError(
                f"With the file path `{self._original_path}` an absolute path relative "
                f"to the calling module `{self._creating_module}` was created, "
                f"resulting `{self._abs_file_path}` - but no file was found."
            )

    @property
    def abs_path(self) -> str:
        if self._abs_file_path != self._original_path:
            logging.debug(
                f"Using abs path `{self._abs_file_path}` for path specified as "
                f"`{self._original_path}` (in `{self._creating_module}`)."
            )
        abs_path = self._abs_file_path
        self._raise_if_not_exists(abs_path)
        return abs_path


class BasetenImage(enum.Enum):
    """Default images, curated by baseten, for different python versions. If a Chainlet
    uses GPUs, drivers will be included in the image."""

    # Enum values correspond to truss canonical python versions.
    PY39 = "py39"
    PY310 = "py310"
    PY311 = "py311"


class CustomImage(SafeModel):
    """Configures the usage of a custom image hosted on dockerhub."""

    image: str
    python_executable_path: Optional[str] = None
    docker_auth: Optional[truss_config.DockerAuthSettings] = None


class DockerImage(SafeModelNonSerializable):
    """Configures the docker image in which a remoted chainlet is deployed.

    Note:
        Any paths are relative to the source file where ``DockerImage`` is
        defined and must be created with the helper function ``make_abs_path_here``.
        This allows you for example organize chainlets in different (potentially nested)
        modules and keep their requirement files right next their python source files.

    Args:
        base_image: The base image used by the chainlet. Other dependencies and
          assets are included as additional layers on top of that image. You can choose
          a Baseten default image for a supported python version (e.g.
          ``BasetenImage.PY311``), this will also include GPU drivers if needed, or
          provide a custom image (e.g. ``CustomImage(image="python:3.11-slim")``)..
        pip_requirements_file: Path to a file containing pip requirements. The file
          content is naively concatenated with ``pip_requirements``.
        pip_requirements: A list of pip requirements to install.  The items are
          naively concatenated with the content of the ``pip_requirements_file``.
        apt_requirements: A list of apt requirements to install.
        data_dir: Data from this directory is copied into the docker image and
          accessible to the remote chainlet at runtime.
        external_package_dirs: A list of directories containing additional python
          packages outside the chain's workspace dir, e.g. a shared library. This code
          is copied into the docker image and importable at runtime.
    """

    # TODO: this is not stable yet and might change or refer back to truss.
    base_image: Union[BasetenImage, CustomImage] = BasetenImage.PY311
    pip_requirements_file: Optional[AbsPath] = None
    pip_requirements: list[str] = []
    apt_requirements: list[str] = []
    data_dir: Optional[AbsPath] = None
    external_package_dirs: Optional[list[AbsPath]] = None

    @pydantic.root_validator(pre=True)
    def migrate_fields(cls, values):
        if "base_image" in values:
            base_image = values["base_image"]
            if isinstance(base_image, str):
                doc_link = "https://docs.baseten.co/chains-reference/sdk#class-truss-chains-dockerimage"
                raise ChainsUsageError(
                    "`DockerImage.base_image` as string is deprecated. Specify as "
                    f"`BasetenImage` or `CustomImage` (see docs: {doc_link})."
                )
        return values


class ComputeSpec(pydantic.BaseModel):
    """Parsed and validated compute.  See ``Compute`` for more information."""

    # TODO: this is not stable yet and might change or refer back to truss.
    cpu_count: int = 1
    predict_concurrency: int = 1
    memory: str = "2Gi"
    accelerator: truss_config.AcceleratorSpec = truss_config.AcceleratorSpec()


CpuCountT = Literal["cpu_count"]
CPU_COUNT: CpuCountT = "cpu_count"


class Compute:
    """Specifies which compute resources a chainlet has in the *remote* deployment.

    Note:
        Not all combinations can be exactly satisfied by available hardware, in some
        cases more powerful machine types are chosen to make sure requirements are met
        or over-provisioned. Refer to the
        `baseten instance reference <https://docs.baseten.co/performance/instances>`_.
    """

    # Builder to create ComputeSpec.
    # This extra layer around `ComputeSpec` is needed to parse the accelerator options.

    _spec: ComputeSpec

    def __init__(
        self,
        cpu_count: int = 1,
        memory: str = "2Gi",
        gpu: Union[str, truss_config.Accelerator, None] = None,
        gpu_count: int = 1,
        predict_concurrency: Union[int, CpuCountT] = 1,
    ) -> None:
        """
        Args:
            cpu_count: Minimum number of CPUs to allocate.
            memory: Minimum memory to allocate, e.g. "2Gi" (2 gibibytes).
            gpu: GPU accelerator type, e.g. "A10G", "A100", refer to the
              `truss config <https://docs.baseten.co/reference/config#resources-accelerator>`_
              for more choices.
            gpu_count: Number of GPUs to allocate.
            predict_concurrency: Number of concurrent requests a single replica of a
              deployed chainlet handles.

        Concurrency concepts are explained in `this guide <https://docs.baseten.co/deploy/guides/concurrency#predict-concurrency>`_. # noqa: E501
        It is important to understand the difference between `predict_concurrency` and
        the concurrency target (used for autoscaling, i.e. adding or removing replicas).
        Furthermore, the ``predict_concurrency`` of a single instance is implemented in
        two ways:

        - Via python's ``asyncio``, if ``run_remote`` is an async def. This
          requires that ``run_remote`` yields to the event loop.

        - With a threadpool if it's a synchronous function. This requires
          that the threads don't have significant CPU load (due to the GIL).
        """
        accelerator = truss_config.AcceleratorSpec()
        if gpu:
            accelerator.accelerator = truss_config.Accelerator(gpu)
            accelerator.count = gpu_count
            accelerator = truss_config.AcceleratorSpec(
                accelerator=truss_config.Accelerator(gpu), count=gpu_count
            )
        if predict_concurrency == CPU_COUNT:
            predict_concurrency_int = cpu_count
        else:
            assert isinstance(predict_concurrency, int)
            predict_concurrency_int = predict_concurrency

        self._spec = ComputeSpec(
            cpu_count=cpu_count,
            memory=memory,
            accelerator=accelerator,
            predict_concurrency=predict_concurrency_int,
        )

    def get_spec(self) -> ComputeSpec:
        return self._spec.model_copy(deep=True)


class AssetSpec(SafeModel):
    """Parsed and validated assets. See ``Assets`` for more information."""

    # TODO: this is not stable yet and might change or refer back to truss.
    secrets: dict[str, str] = pydantic.Field({})
    cached: list[truss_config.ModelRepo] = []
    external_data: list[truss_config.ExternalDataItem] = []


class Assets:
    """Specifies which assets a chainlet can access in the remote deployment.

    For example, model weight caching can be used like this::

        import truss_chains as chains
        from truss.base import truss_config

        mistral_cache = truss_config.ModelRepo(
            repo_id="mistralai/Mistral-7B-Instruct-v0.2",
            allow_patterns=["*.json", "*.safetensors", ".model"]
        )
        chains.Assets(cached=[mistral_cache], ...)

    See `truss caching guide <https://docs.baseten.co/deploy/guides/model-cache#enabling-caching-for-a-model>`_
    for more details on caching.
    """

    # Builder to create asset spec.
    # This extra layer around `AssetSpec` is needed to add secret_keys.
    _spec: AssetSpec

    def __init__(
        self,
        cached: Iterable[truss_config.ModelRepo] = (),
        secret_keys: Iterable[str] = (),
        external_data: Iterable[truss_config.ExternalDataItem] = (),
    ) -> None:
        """
        Args:
            cached: One or more ``truss_config.ModelRepo`` objects.
            secret_keys: Names of secrets stored on baseten, that the
              chainlet should have access to. You can manage secrets on baseten
              `here <https://app.baseten.co/settings/secrets>`_.
            external_data: Data to be downloaded from public URLs and made available
              in the deployment (via ``context.data_dir``). See
              `here <https://docs.baseten.co/reference/config#external-data>`_ for
              more details.
        """
        self._spec = AssetSpec(
            cached=list(cached),
            secrets={k: SECRET_DUMMY for k in secret_keys},
            external_data=list(external_data),
        )

    def get_spec(self) -> AssetSpec:
        """Returns parsed and validated assets."""
        return self._spec.model_copy(deep=True)


class ChainletOptions(SafeModelNonSerializable):
    """
    Args:
        enable_b10_tracing: enables baseten-internal trace data collection. This
          helps baseten engineers better analyze chain performance in case of issues.
          It is independent of a potentially user-configured tracing instrumentation.
          Turning this on, could add performance overhead.
        env_variables: static environment variables available to the deployed chainlet.
    """

    enable_b10_tracing: bool = False
    env_variables: Mapping[str, str] = {}


class RemoteConfig(SafeModelNonSerializable):
    """Bundles config values needed to deploy a chainlet remotely.

    This is specified as a class variable for each chainlet class, e.g.::

            import truss_chains as chains


            class MyChainlet(chains.ChainletBase):
                remote_config = chains.RemoteConfig(
                    docker_image=chains.DockerImage(
                        pip_requirements=["torch==2.0.1", ...]
                    ),
                    compute=chains.Compute(cpu_count=2, gpu="A10G", ...),
                    assets=chains.Assets(secret_keys=["hf_access_token"], ...),
                )

    """

    docker_image: DockerImage = DockerImage()
    compute: Compute = Compute()
    assets: Assets = Assets()
    name: Optional[str] = None
    options: ChainletOptions = ChainletOptions()

    def get_compute_spec(self) -> ComputeSpec:
        return self.compute.get_spec()

    def get_asset_spec(self) -> AssetSpec:
        return self.assets.get_spec()


DEFAULT_TIMEOUT_SEC = 600


class RPCOptions(SafeModel):
    """Options to customize RPCs to dependency chainlets.

    Args:
        retries: The number of times to retry the remote chainlet in case of failures
          (e.g. due to transient network issues). For streaming, retries are only made
          if the request fails before streaming any results back. Failures mid-stream
          not retried.
        timeout_sec: Timeout for the HTTP request to this chainlet.
        use_binary: whether to send data data in binary format. This can give a parsing
         speedup and message size reduction (~25%) for numpy arrays. Use
         ``NumpyArrayField`` as a field type on pydantic models for integration and set
         this option to ``True``. For simple text data, there is no significant benefit.
    """

    retries: int = 1
    timeout_sec: int = DEFAULT_TIMEOUT_SEC
    use_binary: bool = False


class ServiceDescriptor(SafeModel):
    """Bundles values to establish an RPC session to a dependency chainlet,
    specifically with ``StubBase``."""

    name: str
    display_name: str
    options: RPCOptions


class DeployedServiceDescriptor(ServiceDescriptor):
    predict_url: str


class Environment(SafeModel):
    """The environment the chainlet is deployed in.

    Args:
        name: The name of the environment.
    """

    name: str
    # can add more fields here as we add them to dynamic_config configmap


class DeploymentContext(SafeModelNonSerializable):
    """Bundles config values and resources needed to instantiate Chainlets.

    The context can optionally added as a trailing argument in a Chainlet's
    ``__init__`` method and then used to set up the chainlet (e.g. using a secret as
    an access token for downloading model weights).

    Args:
        data_dir: The directory where the chainlet can store and access data,
          e.g. for downloading model weights.
        chainlet_to_service: A mapping from chainlet names to service descriptors.
          This is used create RPCs sessions to dependency chainlets. It contains only
          the chainlet services that are dependencies of the current chainlet.
        secrets: A mapping from secret names to secret values. It contains only the
          secrets that are listed in ``remote_config.assets.secret_keys`` of the
          current chainlet.
        environment: The environment that the chainlet is deployed in.
          None if the chainlet is not associated with an environment.
    """

    data_dir: Optional[pathlib.Path] = None
    chainlet_to_service: Mapping[str, DeployedServiceDescriptor]
    secrets: MappingNoIter[str, str]
    environment: Optional[Environment] = None

    def get_service_descriptor(self, chainlet_name: str) -> DeployedServiceDescriptor:
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
            f"`{BASETEN_API_SECRET_NAME}` on Baseten to allow chain Chainlet to "
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


class TrussMetadata(SafeModel):
    """Plugin for the truss config (in config["model_metadata"]["chains_metadata"])."""

    chainlet_to_service: Mapping[str, ServiceDescriptor]


class ABCChainlet(abc.ABC):
    remote_config: ClassVar[RemoteConfig] = RemoteConfig(docker_image=DockerImage())
    _init_is_patched: ClassVar[bool] = False

    @classmethod
    def has_custom_init(cls) -> bool:
        return cls.__init__ is not object.__init__

    @classproperty
    @classmethod
    def name(cls) -> str:
        return cls.__name__

    @classproperty
    @classmethod
    def display_name(cls) -> str:
        return cls.remote_config.name or cls.name

    # Cannot add this abstract method to API, because we want to allow arbitrary
    # arg/kwarg names and specifying any function signature here would give type errors
    # @abc.abstractmethod
    # def run_remote(self, *args, **kwargs) -> Any:
    #     ...


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


class StreamingTypeDescriptor(TypeDescriptor):
    origin_type: type
    arg_type: type

    @property
    def is_string(self) -> bool:
        return self.arg_type is str

    @property
    def is_pydantic(self) -> bool:
        return False


class InputArg(SafeModelNonSerializable):
    name: str
    type: TypeDescriptor
    is_optional: bool


class EndpointAPIDescriptor(SafeModelNonSerializable):
    name: str = ENDPOINT_METHOD_NAME
    input_args: list[InputArg]
    output_types: list[TypeDescriptor]
    is_async: bool
    is_streaming: bool

    @property
    def streaming_type(self) -> StreamingTypeDescriptor:
        if (
            not self.is_streaming
            or len(self.output_types) != 1
            or not isinstance(self.output_types[0], StreamingTypeDescriptor)
        ):
            raise ValueError(f"{self} is not a streaming endpoint.")
        return cast(StreamingTypeDescriptor, self.output_types[0])


class DependencyDescriptor(SafeModelNonSerializable):
    chainlet_cls: Type[ABCChainlet]
    options: RPCOptions

    @property
    def name(self) -> str:
        return self.chainlet_cls.name

    @property
    def display_name(self) -> str:
        return self.chainlet_cls.display_name


class ChainletAPIDescriptor(SafeModelNonSerializable):
    chainlet_cls: Type[ABCChainlet]
    src_path: str
    has_context: bool
    dependencies: Mapping[str, DependencyDescriptor]
    endpoint: EndpointAPIDescriptor

    def __hash__(self) -> int:
        return hash(self.chainlet_cls)

    @property
    def name(self) -> str:
        return self.chainlet_cls.name

    @property
    def display_name(self) -> str:
        return self.chainlet_cls.display_name


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
    """When a remote chainlet raises an exception, this pydantic model contains
    information about the error and stack trace and is included in JSON form in the
    error response.
    """

    exception_cls_name: str
    exception_module_name: Optional[str]
    exception_message: str
    user_stack_trace: list[StackFrame]

    def _to_stack_summary(self) -> traceback.StackSummary:
        return traceback.StackSummary.from_list(
            frame.to_frame_summary() for frame in self.user_stack_trace
        )

    def format(self) -> str:
        """Format the error for printing, similar to how Python formats exceptions
        with stack traces."""
        stack = "".join(traceback.format_list(self._to_stack_summary()))
        exc_info = (
            f"\n(Exception class defined in `{self.exception_module_name}`.)"
            if self.exception_module_name
            else ""
        )
        error = (
            f"Chainlet-Traceback (most recent call last):\n"
            f"{stack}{self.exception_cls_name}: {self.exception_message}{exc_info}"
        )
        return error


class GenericRemoteException(Exception): ...


########################################################################################


class PushOptions(SafeModelNonSerializable):
    chain_name: str
    only_generate_trusses: bool = False


class PushOptionsBaseten(PushOptions):
    remote_provider: baseten_remote.BasetenRemote
    publish: bool
    environment: Optional[str]

    @classmethod
    def create(
        cls,
        chain_name: str,
        publish: bool,
        promote: Optional[bool],
        only_generate_trusses: bool,
        remote: str,
        environment: Optional[str] = None,
    ) -> "PushOptionsBaseten":
        if promote and not environment:
            environment = PRODUCTION_ENVIRONMENT_NAME
        if environment:
            publish = True
        remote_provider = cast(
            baseten_remote.BasetenRemote,
            remote_factory.RemoteFactory.create(remote=remote),
        )
        return PushOptionsBaseten(
            remote_provider=remote_provider,
            chain_name=chain_name,
            publish=publish,
            only_generate_trusses=only_generate_trusses,
            environment=environment,
        )


class PushOptionsLocalDocker(PushOptions):
    # Local docker-to-docker requests don't need auth, but we need to set a
    # value different from `SECRET_DUMMY` to not trigger the check that the secret
    # is unset. Additionally, if local docker containers make calls to models deployed
    # on baseten, a real API key must be provided (i.e. the default must be overridden).
    baseten_chain_api_key: str = "docker_dummy_key"
    # If enabled, chains code is copied from the local package into `/app/truss_chains`
    # in the docker image (which takes precedence over potential pip/site-packages).
    # This should be used for integration tests or quick local dev loops.
    use_local_chains_src: bool = False
