# TODO: this file contains too much implementation -> restructure.
import abc
import enum
import pathlib
from typing import (  # type: ignore[attr-defined]  # Chains uses Python >=3.9.
    Any,
    Callable,
    ClassVar,
    Generic,
    GenericAlias,  # This causes above type error.
    Mapping,
    Optional,
    Type,
    TypeVar,
    cast,
    get_args,
    get_origin,
)

import pydantic

from truss.base import custom_types
from truss.base.constants import PRODUCTION_ENVIRONMENT_NAME
from truss_chains import public_types, utils

TRUSS_CONFIG_CHAINS_KEY = "chains_metadata"
GENERATED_CODE_DIR = ".chains_generated"
DYNAMIC_CHAINLET_CONFIG_KEY = "dynamic_chainlet_config"
OTEL_TRACE_PARENT_HEADER_KEY = "traceparent"
RUN_REMOTE_METHOD_NAME = "run_remote"  # Chainlet method name exposed as endpoint.
MODEL_ENDPOINT_METHOD_NAME = "predict"  # Model method name exposed as endpoint.
HEALTH_CHECK_METHOD_NAME = "is_healthy"
# Below arg names must correspond to `private_types.ABCChainlet`.
CONTEXT_ARG_NAME = "context"  # Referring to Chainlets `__init__` signature.
SELF_ARG_NAME = "self"
REMOTE_CONFIG_NAME = "remote_config"
ENGINE_BUILDER_CONFIG_NAME = "engine_builder_config"


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


class ChainletMetadata(custom_types.SafeModelNonSerializable):
    is_entrypoint: bool = False
    chain_name: Optional[str] = None
    init_is_patched: bool = False


class EntityType(utils.StrEnum):
    CHAINLET = enum.auto()
    MODEL = enum.auto()
    ENGINE_BUILDER_MODEL = enum.auto()


class FrameworkConfig(custom_types.SafeModelNonSerializable):
    entity_type: EntityType
    supports_dependencies: bool
    endpoint_method_name: str


class ServiceDescriptor(custom_types.SafeModel):
    """like `DeployedServiceDescriptor` but without url."""

    name: str
    display_name: str
    options: public_types.RPCOptions


class TrussMetadata(custom_types.SafeModel):
    """Plugin for the truss config (in config["model_metadata"]["chains_metadata"])."""

    chainlet_to_service: Mapping[str, ServiceDescriptor]


class ABCChainlet(abc.ABC):
    remote_config: ClassVar[public_types.RemoteConfig] = public_types.RemoteConfig()
    # `meta_data` is not shared between subclasses, each has an isolated copy.
    meta_data: ClassVar[ChainletMetadata] = ChainletMetadata()
    _framework_config: ClassVar[FrameworkConfig]

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

    @classproperty
    @classmethod
    def supports_dependencies(cls) -> bool:
        return cls._framework_config.supports_dependencies

    @classproperty
    @classmethod
    def entity_type(cls) -> EntityType:
        return cls._framework_config.entity_type

    @classproperty
    @classmethod
    def endpoint_method_name(cls) -> str:
        return cls._framework_config.endpoint_method_name

    # Cannot add this abstract method to API, because we want to allow arbitrary
    # arg/kwarg names and specifying any function signature here would give type errors
    # @abc.abstractmethod
    # def run_remote(self, *args, **kwargs) -> Any:
    #     ...


class TypeDescriptor(custom_types.SafeModelNonSerializable):
    """For describing I/O types of Chainlets."""

    raw: Any  # The raw type annotation object (could be a type or GenericAlias).

    @property
    def is_pydantic(self) -> bool:
        return (
            isinstance(self.raw, type)
            and not isinstance(self.raw, GenericAlias)
            and issubclass(self.raw, pydantic.BaseModel)
        )

    @property
    def has_pydantic_args(self):
        origin = get_origin(self.raw)
        if not origin:
            return False
        args = get_args(self.raw)
        return any(
            isinstance(arg, type) and issubclass(arg, pydantic.BaseModel)
            for arg in args
        )

    @property
    def is_websocket(self) -> bool:
        return self.raw == public_types.WebSocketProtocol


class StreamingTypeDescriptor(TypeDescriptor):
    origin_type: type
    arg_type: type

    @property
    def is_string(self) -> bool:
        return self.arg_type is str

    @property
    def is_pydantic(self) -> bool:
        return False


class InputArg(custom_types.SafeModelNonSerializable):
    name: str
    type: TypeDescriptor
    is_optional: bool


class EndpointAPIDescriptor(custom_types.SafeModelNonSerializable):
    name: str = RUN_REMOTE_METHOD_NAME
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

    @property
    def is_websocket(self):
        return any(arg.type.is_websocket for arg in self.input_args)

    @property
    def has_pydantic_input(self) -> bool:
        return not self.is_websocket

    @property
    def has_pydantic_output(self) -> bool:
        return not (self.is_streaming or self.is_websocket)

    @property
    def has_engine_builder_llm_input(self) -> bool:
        return any(
            arg.type.raw == public_types.EngineBuilderLLMInput
            for arg in self.input_args
        )


class DependencyDescriptor(custom_types.SafeModelNonSerializable):
    chainlet_cls: Type[ABCChainlet]
    options: public_types.RPCOptions

    @property
    def name(self) -> str:
        return self.chainlet_cls.name

    @property
    def display_name(self) -> str:
        return self.chainlet_cls.display_name


class HealthCheckAPIDescriptor(custom_types.SafeModelNonSerializable):
    name: str = HEALTH_CHECK_METHOD_NAME
    is_async: bool


class ChainletAPIDescriptor(custom_types.SafeModelNonSerializable):
    chainlet_cls: Type[ABCChainlet]
    src_path: str
    has_context: bool
    dependencies: Mapping[str, DependencyDescriptor]
    endpoint: EndpointAPIDescriptor
    health_check: Optional[HealthCheckAPIDescriptor]

    def __hash__(self) -> int:
        return hash(self.chainlet_cls)

    @property
    def name(self) -> str:
        return self.chainlet_cls.name

    @property
    def display_name(self) -> str:
        return self.chainlet_cls.display_name


########################################################################################


class PushOptions(custom_types.SafeModelNonSerializable):
    chain_name: str
    only_generate_trusses: bool = False


class PushOptionsBaseten(PushOptions):
    remote: str
    publish: bool
    environment: Optional[str]
    include_git_info: bool
    working_dir: pathlib.Path

    @classmethod
    def create(
        cls,
        chain_name: str,
        publish: bool,
        promote: Optional[bool],
        only_generate_trusses: bool,
        remote: str,
        include_git_info: bool,
        working_dir: pathlib.Path,
        environment: Optional[str] = None,
    ) -> "PushOptionsBaseten":
        if promote and not environment:
            environment = PRODUCTION_ENVIRONMENT_NAME
        if environment:
            publish = True
        return PushOptionsBaseten(
            remote=remote,
            chain_name=chain_name,
            publish=publish,
            only_generate_trusses=only_generate_trusses,
            environment=environment,
            include_git_info=include_git_info,
            working_dir=working_dir,
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
    use_local_src: bool = False
