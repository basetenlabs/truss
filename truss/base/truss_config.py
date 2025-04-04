import enum
import logging
import math
import os
import pathlib
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, ClassVar, Mapping, MutableMapping, Optional

import pydantic
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


def _is_numeric(number_like: str) -> bool:
    try:
        float(number_like)
        return True
    except ValueError:
        return False


class Accelerator(str, enum.Enum):
    T4 = "T4"
    L4 = "L4"
    A10G = "A10G"
    V100 = "V100"
    A100 = "A100"
    A100_40GB = "A100_40GB"
    H100 = "H100"
    H200 = "H200"
    H100_40GB = "H100_40GB"


class AcceleratorSpec(custom_types.ConfigModel):
    accelerator: Optional[Accelerator] = None
    count: int = pydantic.Field(default=1, ge=0)

    def _to_string_spec(self) -> Optional[str]:
        if self.accelerator is None or self.count <= 0:
            return None
        elif self.count > 1:
            return f"{self.accelerator.value}:{self.count}"
        return self.accelerator.value

    @classmethod
    def _from_string_spec(cls, value: str) -> "AcceleratorSpec":
        parts = value.strip().split(":")
        if not parts[0]:
            raise ValueError("Accelerator type cannot be empty.")
        if len(parts) > 2:
            raise ValueError("Expected format: `Accelerator` or `Accelerator:count`.")
        try:
            acc = Accelerator(parts[0])
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
        return cls(accelerator=acc, count=count)

    @classmethod
    def _validate_input(
        cls,
        value: Any,
        handler: core_schema.ValidatorFunctionWrapHandler,
        info: core_schema.ValidationInfo,
    ) -> "AcceleratorSpec":
        if isinstance(value, str):
            return cls._from_string_spec(value)
        if isinstance(value, dict):
            return handler(value)
        if isinstance(value, cls):
            return value
        if value is None:
            return cls()
        raise TypeError(
            f"Expected string, dict, None or AcceleratorSpec; got {type(value)}."
        )

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: pydantic.GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        # Hooks up custom parsing and serialization with pydantic.
        schema = core_schema.with_info_wrap_validator_function(
            cls._validate_input, handler(source_type)
        )
        return core_schema.json_or_python_schema(
            json_schema=schema,
            python_schema=schema,
            serialization=core_schema.plain_serializer_function_ser_schema(
                function=cls._to_string_spec,
                info_arg=False,
                return_schema=core_schema.str_schema(),
                when_used="json-unless-none",
            ),
        )

    @classmethod
    def __get_pydantic_json_schema__(
        cls,
        core_schema_obj: core_schema.CoreSchema,
        handler: pydantic.GetJsonSchemaHandler,
    ) -> json_schema.JsonSchemaValue:
        # Hooks up differing JSON schema from python fields with pydantic.
        schema = handler(core_schema_obj)
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
    def runtime_path(self) -> "Path":
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
        if v.get("volume_folder") is None:
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
        return len(self.models) >= 1 and any(
            model.use_volume is True for model in self.models
        )

    def _check_volume_consistent(self):
        """Check if all models have the same volume folder."""
        if len(self.models) == 0:
            return
        if not all(
            model.volume_folder == self.models[0].volume_folder for model in self.models
        ):
            raise ValueError(
                "All models in the `model_cache` must either use `use_volume=True` or `use_volume=False`."
            )


class CacheInternal(ModelCache): ...


class HealthChecks(custom_types.ConfigModel):
    restart_check_delay_seconds: Optional[int] = None
    restart_threshold_seconds: Optional[int] = None
    stop_traffic_threshold_seconds: Optional[int] = None


class Runtime(custom_types.ConfigModel):
    predict_concurrency: int = 1
    streaming_read_timeout: int = 60
    enable_tracing_data: bool = False
    enable_debug_logs: bool = False
    is_websocket_endpoint: bool = False
    health_checks: HealthChecks = pydantic.Field(default_factory=HealthChecks)

    @pydantic.model_validator(mode="before")
    def _check_legacy_workers(cls, values: dict) -> dict:
        if "num_workers" in values and values["num_workers"] != 1:
            raise ValueError(
                "After truss 0.9.49 only 1 worker per server is allowed. "
                "For concurrency utilize asyncio, autoscaling replicas "
                "and as a last resort thread/process pools inside the truss model."
            )
        return values


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


class DockerAuthSettings(custom_types.ConfigModel):
    """Provides information about how to authenticate to the docker registry containing
    the custom base image."""

    auth_method: DockerAuthType
    secret_name: str
    registry: Optional[str] = ""

    @pydantic.field_validator("auth_method", mode="before")
    def _normalize_auth_method(cls, v: str) -> str:
        return v.upper() if isinstance(v, str) else v


class BaseImage(custom_types.ConfigModel):
    image: str = ""
    python_executable_path: str = ""
    docker_auth: Optional[DockerAuthSettings] = None

    @pydantic.field_validator("python_executable_path")
    def _validate_path(cls, v: str) -> str:
        if v and not PurePosixPath(v).is_absolute():
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


SUPPORTED_PYTHON_VERSIONS = {
    "py38": "3.8",
    "py39": "3.9",
    "py310": "3.10",
    "py311": "3.11",
}


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
        return SUPPORTED_PYTHON_VERSIONS[self.python_version]

    @property
    def parsed_trt_llm_build_configs(
        self,
    ) -> list[trt_llm_config.TrussTRTLLMBuildConfiguration]:
        if self.trt_llm:
            if self.trt_llm.build.speculator and self.trt_llm.build.speculator.build:
                return [self.trt_llm.build, self.trt_llm.build.speculator.build]
            return [self.trt_llm.build]
        return []

    def to_dict(self, verbose: bool = True) -> dict:
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
        valid = ["py38", "py39", "py310", "py311"]
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

    Currently, it maps any versions greater than 3.11 to 3.11.

    Args:
        python_version: in the form py[major_version][minor_version] e.g. py39,
          py310.
    """
    python_major_version = int(python_version[2:3])
    python_minor_version = int(python_version[3:])

    if python_major_version != 3:
        raise NotImplementedError("Only python version 3 is supported")

    if python_minor_version > 11:
        logger.info(
            f"Mapping python version {python_major_version}.{python_minor_version}"
            " to 3.11, the highest version that Truss currently supports."
        )
        return "py311"

    if python_minor_version < 8:
        raise ValueError(
            f"Mapping python version {python_major_version}.{python_minor_version}"
            " to 3.8, the lowest version that Truss currently supports."
        )

    return python_version


def map_local_to_supported_python_version() -> str:
    return _map_to_supported_python_version(
        f"py{sys.version_info.major}{sys.version_info.minor}"
    )
