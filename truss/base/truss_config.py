import enum
import logging
import math
import os
import pathlib
import re
import sys
import warnings
from typing import (
    Annotated,
    Any,
    ClassVar,
    Literal,
    Mapping,
    MutableMapping,
    Optional,
    Union,
)

import packaging.version
import pydantic
import pydantic_core
import yaml
from pydantic import json_schema
from pydantic_core import core_schema

from truss.base import constants, custom_types, trt_llm_config
from truss.util.requirements import parse_requirement_string, raise_insufficent_revision

logger = logging.getLogger(__name__)

TRTLLMConfiguration = trt_llm_config.TRTLLMConfiguration  # Export as alias.
DEFAULT_MODEL_MODULE_DIR = "model"
DEFAULT_BUNDLED_PACKAGES_DIR = "packages"
DEFAULT_DATA_DIRECTORY = "data"
DEFAULT_CPU = "1"
DEFAULT_MEMORY = "2Gi"
DEFAULT_AWS_ACCESS_KEY_SECRET_NAME = "aws_access_key_id"
DEFAULT_AWS_SECRET_ACCESS_KEY_SECRET_NAME = "aws_secret_access_key"

DEFAULT_TRAINING_CHECKPOINT_FOLDER = "/tmp/training_checkpoints"


def _is_numeric(number_like: str) -> bool:
    try:
        float(number_like)
        return True
    except ValueError:
        return False


class Accelerator(str, enum.Enum):
    _B10 = "B10"
    T4 = "T4"
    L4 = "L4"
    A10G = "A10G"
    V100 = "V100"
    A100 = "A100"
    A100_40GB = "A100_40GB"
    H100 = "H100"
    H200 = "H200"
    H100_40GB = "H100_40GB"
    B200 = "B200"


class AcceleratorSpec(custom_types.ConfigModel):
    model_config = pydantic.ConfigDict(validate_assignment=True)

    accelerator: Optional[Accelerator] = None
    count: int = pydantic.Field(default=1, ge=0)

    @classmethod
    def _from_string_spec(cls, value: str) -> dict:
        parts = value.strip().split(":")
        if not parts[0]:
            raise ValueError("Accelerator type cannot be empty.")
        try:
            accelerator = Accelerator(parts[0])
        except ValueError:
            available = ", ".join(a.value for a in Accelerator)
            raise ValueError(
                f"Unsupported accelerator type: `{parts[0]}`. Available types: {available}."
            )
        count = 1
        if len(parts) == 2:
            if not parts[1].isdigit() or int(parts[1]) <= 0:
                raise ValueError(
                    f"Invalid count: '{parts[1]}'. Must be positive integer."
                )
            count = int(parts[1])
        return {"accelerator": accelerator, "count": count}

    @pydantic.model_validator(mode="before")
    @classmethod
    def _parse_combined_spec(cls, value: object) -> object:
        if isinstance(value, str):
            return cls._from_string_spec(value)
        if isinstance(value, AcceleratorSpec):
            return value.dict()
        if isinstance(value, dict):
            return value
        if value is None:
            return {}
        raise TypeError(
            f"Expected string, dict, AcceleratorSpec, or None; got {type(value)}"
        )

    @pydantic.model_serializer(mode="plain")
    def _to_string_spec(self) -> Optional[str]:
        if self.accelerator is None or self.count <= 0:
            return None
        if self.count > 1:
            return f"{self.accelerator.value}:{self.count}"
        return self.accelerator.value

    @classmethod
    def model_json_schema(  # type: ignore[override]
        cls,
        core_schema: pydantic_core.CoreSchema,
        handler: pydantic.GetJsonSchemaHandler,
    ) -> json_schema.JsonSchemaValue:
        schema = handler(core_schema)
        schema.update(
            type="string",
            examples=["A100", "T4:2", "H100:8"],
            description="Accelerator specification in 'TYPE' or 'TYPE:count' format.",
        )
        schema.pop("properties", None)
        schema.pop("required", None)
        return schema


class ModelRepo(custom_types.ConfigModel):
    repo_id: Annotated[str, pydantic.StringConstraints(min_length=1)]
    revision: Optional[Annotated[str, pydantic.StringConstraints(min_length=1)]] = None
    allow_patterns: Optional[list[str]] = None
    ignore_patterns: Optional[list[str]] = None
    volume_folder: Optional[
        Annotated[str, pydantic.StringConstraints(min_length=1)]
    ] = None
    use_volume: bool = False

    @property
    def runtime_path(self) -> pathlib.Path:
        assert self.volume_folder is not None
        return constants.MODEL_CACHE_PATH / self.volume_folder

    @pydantic.model_validator(mode="before")
    def _check_v2_requirements(cls, v) -> str:
        use_volume = v.get("use_volume", False)
        if not use_volume:
            return v
        if v.get("revision") is None:
            logger.warning(
                "the key `revision: str` is required for use_volume=True repos."
            )
            raise_insufficent_revision(v.get("repo_id"), v.get("revision"))
        if v.get("volume_folder") is None or len(v["volume_folder"]) == 0:
            raise ValueError(
                "the key `volume_folder: str` is required for `use_volume=True` repos."
            )
        return v


class ModelCache(pydantic.RootModel[list[ModelRepo]]):
    @property
    def models(self) -> list[ModelRepo]:
        return self.root

    @property
    def is_v1(self) -> bool:
        self._check_volume_consistent()
        return len(self.models) >= 1 and all(
            model.use_volume is False for model in self.models
        )

    @property
    def is_v2(self) -> bool:
        self._check_volume_consistent()
        return len(self.models) >= 1 and any(model.use_volume for model in self.models)

    def _check_volume_consistent(self):
        """Check if all models have the same volume folder."""
        if len(self.models) == 0:
            return
        if not all(
            model.use_volume == self.models[0].use_volume for model in self.models
        ):
            raise ValueError(
                "All models in the `model_cache` must either use `use_volume=True` or `use_volume=False`."
            )


class CacheInternal(ModelCache): ...


class HealthChecks(custom_types.ConfigModel):
    restart_check_delay_seconds: Optional[int] = None
    restart_threshold_seconds: Optional[int] = None
    stop_traffic_threshold_seconds: Optional[int] = None


class TransportKind(str, enum.Enum):
    HTTP = "http"
    WEBSOCKET = "websocket"
    GRPC = "grpc"


class HTTPOptions(pydantic.BaseModel):
    kind: Literal["http"] = "http"


class WebsocketOptions(pydantic.BaseModel):
    kind: Literal["websocket"] = "websocket"
    ping_interval_seconds: Optional[float] = None
    ping_timeout_seconds: Optional[float] = None


class GRPCOptions(pydantic.BaseModel):
    kind: Literal["grpc"] = "grpc"


Transport = Annotated[
    Union[HTTPOptions, WebsocketOptions, GRPCOptions],
    pydantic.Field(discriminator="kind"),
]


class Runtime(custom_types.ConfigModel):
    predict_concurrency: int = 1
    streaming_read_timeout: int = 60
    enable_tracing_data: bool = False
    enable_debug_logs: bool = False
    transport: Transport = HTTPOptions()
    is_websocket_endpoint: Optional[bool] = pydantic.Field(
        None,
        description="DEPRECATED. Do not set manually. Automatically inferred from `transport.kind == websocket`.",
    )
    health_checks: HealthChecks = pydantic.Field(default_factory=HealthChecks)
    truss_server_version_override: Optional[str] = pydantic.Field(
        None,
        description="By default, truss servers are built from the same release as the "
        "CLI used to push. This field allows specifying a pinned/specific version instead.",
    )

    config: ClassVar = pydantic.ConfigDict(validate_assignment=True)

    @pydantic.model_validator(mode="before")
    def _check_legacy_workers(cls, values: dict) -> dict:
        if "num_workers" in values and values["num_workers"] != 1:
            raise ValueError(
                "After truss 0.9.49 only 1 worker per server is allowed. "
                "For concurrency utilize asyncio, autoscaling replicas "
                "and as a last resort thread/process pools inside the truss model."
            )
        return values

    @pydantic.model_validator(mode="before")
    def _handle_legacy_input(cls, values: dict) -> dict:
        if values.get("transport") is None and values.get("is_websocket_endpoint"):
            warnings.warn(
                "`is_websocket_endpoint` is deprecated, use `transport.kind == websocket`",
                DeprecationWarning,
            )
            values["transport"] = {"kind": "websocket"}
        return values

    @pydantic.model_validator(mode="after")
    def sync_is_websocket(self) -> "Runtime":
        transport = self.transport
        if self.is_websocket_endpoint and transport.kind != TransportKind.WEBSOCKET:
            transport = WebsocketOptions()

        is_websocket_endpoint = transport.kind == TransportKind.WEBSOCKET

        # Only update if values actually change and bypass validation to avoid inifite
        # recursion.
        if transport != self.transport:
            object.__setattr__(self, "transport", transport)
        if is_websocket_endpoint != self.is_websocket_endpoint:
            object.__setattr__(self, "is_websocket_endpoint", is_websocket_endpoint)

        return self

    @pydantic.field_validator("transport", mode="before")
    def _default_transport_kind(cls, value: Any) -> Any:
        if value == {}:
            return {"kind": "http"}
        return value

    @pydantic.field_validator("truss_server_version_override")
    def _validate_semver(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        try:
            packaging.version.Version(value)
        except packaging.version.InvalidVersion as e:
            raise ValueError(
                f"Invalid version string: `{value}` - must be parsable as semver, e.g. '0.9.0'."
            ) from e
        return value

    @pydantic.model_serializer(mode="wrap")
    def _serialize_transport(self, serializer):
        data = serializer(self)
        data["transport"] = self.transport.model_dump(exclude_defaults=False)
        return data


class ModelServer(str, enum.Enum):
    """To determine the image builder path for trusses built from alternative server backends.
    This enum is also used to gate development deployments to BasetenRemote
    https://github.com/basetenlabs/truss/blob/7505c17a2ddd4a6fa626b9126772999dc8f3fa86/truss/remote/baseten/remote.py#L56-L57"""

    TrussServer = "TrussServer"
    TRT_LLM = "TRT_LLM"


class Build(custom_types.ConfigModel):
    model_server: ModelServer = ModelServer.TrussServer
    arguments: dict[str, Any] = pydantic.Field(default_factory=dict)
    secret_to_path_mapping: Mapping[str, str] = pydantic.Field(default_factory=dict)

    _SECRET_NAME_REGEX: ClassVar[re.Pattern] = re.compile(r"^[-._a-zA-Z0-9]+$")
    _MAX_SECRET_NAME_LENGTH: ClassVar[int] = 253

    class Config:
        protected_namespaces = ()  # Silence warnings about fields starting with `model_`.

    @classmethod
    def validate_secret_name(cls, secret_name: str) -> None:
        if not isinstance(secret_name, str) or not secret_name:
            raise ValueError(f"Invalid secret name `{secret_name}`")
        if len(secret_name) > cls._MAX_SECRET_NAME_LENGTH:
            raise ValueError(
                f"Secret name `{secret_name}` must be shorter than {cls._MAX_SECRET_NAME_LENGTH}."
            )
        if secret_name in {".", ".."}:
            raise ValueError(f"Secret name `{secret_name}` cannot be `{secret_name}`.")

        if not cls._SECRET_NAME_REGEX.match(secret_name):
            if not secret_name.startswith(constants.REGISTRY_BUILD_SECRET_PREFIX):
                raise ValueError(
                    f"Secrets used in builds must follow Kubernetes object naming "
                    f"conventions. Name `{secret_name}` is not valid. Please comply "
                    f"with the regex `{cls._SECRET_NAME_REGEX.pattern}` and do not start "
                    f"with `{constants.REGISTRY_BUILD_SECRET_PREFIX}`."
                )

    @pydantic.model_validator(mode="after")
    def _validate_secrets(self) -> "Build":
        for secret_name, path in self.secret_to_path_mapping.items():
            self.validate_secret_name(secret_name)
        return self


class Resources(custom_types.ConfigModel):
    cpu: str = DEFAULT_CPU
    memory: str = DEFAULT_MEMORY
    accelerator: AcceleratorSpec = pydantic.Field(default_factory=AcceleratorSpec)
    node_count: Optional[Annotated[int, pydantic.Field(ge=1, strict=True)]] = None

    _MILLI_CPU_REGEX: ClassVar[re.Pattern] = re.compile(r"^[0-9.]*m$")
    _MEMORY_REGEX: ClassVar[re.Pattern] = re.compile(r"^[0-9.]*([a-zA-Z]+)?$")
    _MEMORY_UNITS: ClassVar[dict[str, int]] = {
        "k": 10**3,
        "M": 10**6,
        "G": 10**9,
        "T": 10**12,
        "P": 10**15,
        "E": 10**18,
        "Ki": 1024,
        "Mi": 1024**2,
        "Gi": 1024**3,
        "Ti": 1024**4,
        "Pi": 1024**5,
        "Ei": 1024**6,
    }

    @pydantic.computed_field(return_type=bool)
    def use_gpu(self) -> bool:
        return self.accelerator.accelerator is not None

    @property
    def memory_in_bytes(self) -> int:
        if _is_numeric(self.memory):
            return math.ceil(float(self.memory))

        match = self._MEMORY_REGEX.search(self.memory)
        assert match
        unit = match.group(1)
        return math.ceil(float(self.memory.strip(unit)) * self._MEMORY_UNITS[unit])

    @pydantic.model_validator(mode="before")
    @classmethod
    def strip_use_gpu(cls, data: Any) -> Any:
        # We want `use_gpu` to be serialized, but don't allow extra inputs when parsing,
        # so we have to drop it here to allow roundtrips.
        if isinstance(data, dict):
            data = data.copy()
            data.pop("use_gpu", None)
        return data

    @pydantic.field_validator("cpu")
    def _validate_cpu(cls, cpu_spec: str) -> str:
        if _is_numeric(cpu_spec):
            return cpu_spec

        if not cls._MILLI_CPU_REGEX.fullmatch(cpu_spec):
            raise ValueError(f"Invalid cpu specification {cpu_spec}.")
        return cpu_spec

    @pydantic.field_validator("memory")
    def _validate_memory(cls, mem_spec: str) -> str:
        if _is_numeric(mem_spec):
            return mem_spec

        match = cls._MEMORY_REGEX.fullmatch(mem_spec)
        if not match:
            raise ValueError(f"Invalid memory specification {mem_spec}")

        unit = match.group(1)
        if unit not in cls._MEMORY_UNITS:
            raise ValueError(f"Invalid memory unit {unit} in {mem_spec}")

        return mem_spec

    @pydantic.model_serializer(mode="wrap")
    def _serialize(
        self,
        handler: core_schema.SerializerFunctionWrapHandler,
        info: core_schema.SerializationInfo,
    ) -> dict:
        """Custom omission of `node_count` if at default."""
        result = handler(self)
        if not self.node_count:
            result.pop("node_count", None)
        return result


class ExternalDataItem(custom_types.ConfigModel):
    """A piece of remote data, to be made available to the Truss at serving time.

    Remote data is downloaded and stored under Truss's data directory. Care should be taken
    to avoid conflicts. This will get precedence if there's overlap.
    """

    url: Annotated[str, pydantic.StringConstraints(min_length=1)] = pydantic.Field(
        ...,
        description="URL to download the data from. Currently only files are allowed.",
    )
    local_data_path: Annotated[str, pydantic.StringConstraints(min_length=1)] = (
        pydantic.Field(
            ...,
            description="Path relative to the data directory where the remote file will be downloaded.",
        )
    )
    backend: str = pydantic.Field(
        default="http_public",
        description="Download backend to use. Defaults to 'http_public'.",
    )
    name: Optional[str] = pydantic.Field(
        default=None,
        description="Optional name for the download. Path relative to data directory.",
    )


class ExternalData(pydantic.RootModel[list[ExternalDataItem]]):
    """[Experimental] External data is data that is not contained in the Truss folder.

    Typically, this will be data stored remotely. This data is guaranteed to be made
    available under the data directory of the truss."""

    @property
    def items(self) -> list[ExternalDataItem]:
        return self.root


class DockerAuthType(str, enum.Enum):
    """This enum will express all the types of registry
    authentication we support."""

    GCP_SERVICE_ACCOUNT_JSON = "GCP_SERVICE_ACCOUNT_JSON"
    AWS_IAM = "AWS_IAM"


class DockerAuthSettings(custom_types.ConfigModel):
    """Provides information about how to authenticate to the docker registry containing
    the custom base image."""

    auth_method: DockerAuthType
    registry: Optional[str] = ""
    # Note that the secret_name is only required for GCP_SERVICE_ACCOUNT_JSON.
    secret_name: Optional[str] = None

    # These are only required for AWS_IAM, and only need to be
    # provided in cases where the user wants to use different secret
    # names for the AWS credentials.
    aws_access_key_id_secret_name: str = DEFAULT_AWS_ACCESS_KEY_SECRET_NAME
    aws_secret_access_key_secret_name: str = DEFAULT_AWS_SECRET_ACCESS_KEY_SECRET_NAME

    @pydantic.field_validator("auth_method", mode="before")
    def _normalize_auth_method(cls, v: str) -> str:
        return v.upper() if isinstance(v, str) else v

    @pydantic.model_validator(mode="after")
    def validate_secret_name(self) -> "DockerAuthSettings":
        if (
            self.auth_method == DockerAuthType.GCP_SERVICE_ACCOUNT_JSON
            and self.secret_name is None
        ):
            raise ValueError(
                "secret_name must be provided when auth_method is GCP_SERVICE_ACCOUNT_JSON"
            )
        return self


class BaseImage(custom_types.ConfigModel):
    image: str = ""
    python_executable_path: str = ""
    docker_auth: Optional[DockerAuthSettings] = None

    @pydantic.field_validator("python_executable_path")
    def _validate_path(cls, v: str) -> str:
        if v and not pathlib.PurePosixPath(v).is_absolute():
            raise ValueError(
                f"Invalid relative python executable path {v}. Provide an absolute path"
            )
        return v


class DockerServer(custom_types.ConfigModel):
    start_command: str
    server_port: int
    predict_endpoint: str
    readiness_endpoint: str
    liveness_endpoint: str


class Checkpoint(custom_types.ConfigModel):
    # NB(rcano): The id here is a formatted string of the form <training_job_id>/<checkpoint_id>
    # We do this because the vLLM command requires knowledge of where the checkpoint
    # is downloaded. By using a formatted string instead of an additional "training_job_id"
    # field, we provide a more transparent truss config.
    id: str
    name: str


class CheckpointList(custom_types.ConfigModel):
    download_folder: str = DEFAULT_TRAINING_CHECKPOINT_FOLDER
    checkpoints: list[Checkpoint] = pydantic.Field(default_factory=list)


# TODO: remove just use normal python version instead of this.
def to_dotted_python_version(truss_python_version: str) -> str:
    """Converts python version string using in truss config to the conventional dotted form.

    e.g. py39 to 3.9
    """
    return f"{truss_python_version[2]}.{truss_python_version[3:]}"


class TrussConfig(custom_types.ConfigModel):
    model_name: Optional[str] = None
    model_metadata: dict[str, Any] = pydantic.Field(default_factory=dict)
    description: Optional[str] = None
    examples_filename: str = "examples.yaml"

    data_dir: str = DEFAULT_DATA_DIRECTORY
    external_data: Optional[ExternalData] = None
    external_package_dirs: list[str] = pydantic.Field(default_factory=list)

    python_version: str = "py39"
    base_image: Optional[BaseImage] = None
    requirements_file: Optional[str] = None
    requirements: list[str] = pydantic.Field(default_factory=list)
    system_packages: list[str] = pydantic.Field(default_factory=list)
    environment_variables: dict[str, str] = pydantic.Field(default_factory=dict)
    secrets: MutableMapping[str, Optional[str]] = pydantic.Field(default_factory=dict)

    resources: Resources = pydantic.Field(default_factory=Resources)
    runtime: Runtime = pydantic.Field(default_factory=Runtime)
    build: Build = pydantic.Field(default_factory=Build)
    build_commands: list[str] = pydantic.Field(default_factory=list)
    docker_server: Optional[DockerServer] = None
    model_cache: ModelCache = pydantic.Field(default_factory=lambda: ModelCache([]))
    trt_llm: Optional[trt_llm_config.TRTLLMConfiguration] = None

    # deploying from checkpoint
    buildless_deploy: Optional[bool] = None
    training_checkpoints: Optional[CheckpointList] = None

    # Internal / Legacy.
    input_type: str = "Any"
    model_framework: str = "custom"
    model_type: str = "Model"
    model_module_dir: str = DEFAULT_MODEL_MODULE_DIR
    model_class_filename: str = "model.py"
    model_class_name: str = "Model"
    bundled_packages_dir: str = DEFAULT_BUNDLED_PACKAGES_DIR
    use_local_src: bool = False
    cache_internal: CacheInternal = pydantic.Field(
        default_factory=lambda: CacheInternal([])
    )
    live_reload: bool = False
    apply_library_patches: bool = True
    spec_version: str = "2.0"

    class Config:
        protected_namespaces = ()  # Silence warnings about fields starting with `model_`.

    @property
    def canonical_python_version(self) -> str:
        return to_dotted_python_version(self.python_version)

    def to_dict(self, verbose: bool = True) -> dict:
        self.runtime.sync_is_websocket()  # type: ignore[operator]  # This is callable.
        data = super().to_dict(verbose)
        # Always include.
        data["resources"] = self.resources.to_dict(verbose=True)
        data["python_version"] = self.python_version
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "TrussConfig":
        if "hf_cache" in data:
            logger.warning(
                "Warning: `hf_cache` is deprecated in favor of `model_cache`. "
                "Everything will run as before, but if you are pulling weights from S3 "
                "or GCS, they will be stored at /app/model_cache instead of "
                "/app/hf_cache as before."
            )
        if "hf_cache" in data and "model_cache" not in data:
            data["model_cache"] = data.pop("hf_cache") or []
        data["environment_variables"] = {
            k: str(v).lower() if isinstance(v, bool) else str(v)
            for k, v in data.get("environment_variables", {}).items()
        }
        return cls.model_validate(data)

    @classmethod
    def from_yaml(cls, path: pathlib.Path) -> "TrussConfig":
        if not os.path.isfile(path):
            raise ValueError(f"Expected a truss configuration file at {path}")
        with path.open() as f:
            raw_data = yaml.safe_load(f) or {}
        return cls.from_dict(raw_data)

    def write_to_yaml_file(self, path: pathlib.Path, verbose: bool = True):
        with path.open("w") as config_file:
            yaml.safe_dump(self.to_dict(verbose=verbose), config_file)

    def clone(self) -> "TrussConfig":
        return self.from_dict(self.to_dict())

    def load_requirements_from_file(self, truss_dir: pathlib.Path) -> list[str]:
        if self.requirements_file:
            requirements_path = truss_dir / self.requirements_file
            try:
                requirements = []
                with open(requirements_path) as f:
                    for line in f.readlines():
                        parsed_line = parse_requirement_string(line)
                        if parsed_line:
                            requirements.append(parsed_line)
                return requirements
            except Exception as e:
                logger.exception(
                    f"failed to read requirements file: {self.requirements_file}"
                )
                raise e
        return []

    @staticmethod
    def load_requirements_file_from_filepath(yaml_path: pathlib.Path) -> list[str]:
        config = TrussConfig.from_yaml(yaml_path)
        return config.load_requirements_from_file(yaml_path.parent)

    @pydantic.field_validator("python_version")
    def _validate_python_version(cls, v: str) -> str:
        valid = {f"py{x.replace('.', '')}" for x in constants.SUPPORTED_PYTHON_VERSIONS}
        if v not in valid:
            raise ValueError(f"Please ensure that `python_version` is one of {valid}")
        return v

    @pydantic.model_validator(mode="after")
    def _validate_config(self) -> "TrussConfig":
        if self.requirements and self.requirements_file:
            raise ValueError(
                "Please ensure that only one of `requirements` and `requirements_file` is specified"
            )
        return self

    @pydantic.field_validator("cache_internal", mode="before")
    def _default_cache_internal_if_none(cls, v: Any) -> CacheInternal:
        return CacheInternal([]) if v is None else v

    @pydantic.model_validator(mode="after")
    def _validate_trt_llm_resources(self) -> "TrussConfig":
        return trt_llm_config.trt_llm_validation(self)

    @pydantic.field_serializer("trt_llm")
    def _serialize_trt_llm(
        self,
        trt_llm: Optional[trt_llm_config.TRTLLMConfiguration],
        info: core_schema.FieldSerializationInfo,
    ) -> Optional[dict[str, Any]]:
        if not trt_llm:
            return None
        exclude_unset = bool(info.context and "verbose" in info.context)
        return trt_llm.model_dump(exclude_unset=exclude_unset)


def _map_to_supported_python_version(python_version: str) -> str:
    """Map python version to truss supported python version.

    Currently, it maps any versions greater than max supported version to max.

    Args:
        python_version: in the form py[major_version][minor_version] e.g. py39,
          py310.
    """
    python_major_version = int(python_version[2:3])
    python_minor_version = int(python_version[3:])

    max_minor = packaging.version.parse(constants.SUPPORTED_PYTHON_VERSIONS[-1]).minor
    min_minor = packaging.version.parse(constants.SUPPORTED_PYTHON_VERSIONS[0]).minor

    if python_major_version != 3:
        raise NotImplementedError("Only python version 3 is supported")

    if python_minor_version > max_minor:
        logger.info(
            f"Mapping python version {python_major_version}.{python_minor_version}"
            f" to {constants.SUPPORTED_PYTHON_VERSIONS[-1]}, the highest version that Truss currently supports."
        )
        return f"py{constants.SUPPORTED_PYTHON_VERSIONS[-1].replace('.', '')}"

    if python_minor_version < min_minor:
        raise ValueError(
            f"Mapping python version {python_major_version}.{python_minor_version}"
            f" to {constants.SUPPORTED_PYTHON_VERSIONS[0]}, the lowest version that Truss currently supports."
        )

    return python_version


def map_local_to_supported_python_version() -> str:
    return _map_to_supported_python_version(
        f"py{sys.version_info.major}{sys.version_info.minor}"
    )
