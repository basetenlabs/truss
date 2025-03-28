import enum
import logging
import math
import os
import pathlib
import re
import sys
from pathlib import PurePosixPath
from typing import Annotated, Any, ClassVar, Optional

import pydantic
import yaml
from pydantic import json_schema
from pydantic_core import core_schema

from truss.base import custom_types, trt_llm_config
from truss.base.constants import REGISTRY_BUILD_SECRET_PREFIX
from truss.base.errors import ValidationError
from truss.util.requirements import parse_requirement_string

logger = logging.getLogger(__name__)

TRTLLMConfiguration = trt_llm_config.TRTLLMConfiguration  # Export as alias.
DEFAULT_MODEL_MODULE_DIR = "model"
DEFAULT_BUNDLED_PACKAGES_DIR = "packages"
DEFAULT_DATA_DIRECTORY = "data"
DEFAULT_CPU = "1"
DEFAULT_MEMORY = "2Gi"
DEFAULT_USE_GPU = False


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

    def _to_string(self) -> Optional[str]:
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
        raise ValueError(
            f"Expected string, dict, or AcceleratorSpec; got {type(value)}."
        )

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: pydantic.GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        """
        Defines how Pydantic should validate and serialize AcceleratorSpec.
        Uses a wrap validator to handle string/dict/instance inputs and
        a plain serializer to output the string format.
        """
        # Get the default schema Pydantic would generate for validation (handles dicts)
        default_schema = handler(source_type)

        # Define a schema using a wrap validator (_validate_input)
        # This validator will try parsing strings and delegate dicts/instances back to Pydantic's default handling
        validation_schema = core_schema.with_info_wrap_validator_function(
            cls._validate_input, default_schema
        )

        # Define the serialization logic (instance -> string)
        serialization_schema = core_schema.plain_serializer_function_ser_schema(  # Correct function name
            function=cls._to_string,
            info_arg=False,
            return_schema=core_schema.str_schema(),  # Use return_schema with a schema object
            when_used="json-unless-none",
        )

        return core_schema.json_or_python_schema(
            # Use the custom validation logic for incoming JSON/Python data
            json_schema=validation_schema,
            python_schema=validation_schema,  # Same logic for Python dicts/values
            # Apply the custom serialization logic when dumping the model
            serialization=serialization_schema,
            # Link back to the default schema for other Pydantic internals if needed
            # (might not be strictly necessary with wrap validator but good practice)
            # core_schema=default_schema # This might re-introduce recursion issues depending on exact Pydantic version/use. Often safer without it when using wrap.
        )

    @classmethod
    def __get_pydantic_json_schema__(
        cls,
        core_schema_obj: core_schema.CoreSchema,
        handler: pydantic.GetJsonSchemaHandler,
    ) -> json_schema.JsonSchemaValue:
        """
        Overrides the JSON schema generation to represent this type as a string.
        """
        # Get the schema Pydantic would normally generate based on the core_schema
        # (which includes our validation/serialization but might still resolve to an object internally)
        schema = handler(core_schema_obj)

        # Override the type and add examples reflecting the string format
        schema.update(
            type="string",
            examples=["A100", "T4:2", "H100:8"],
            description="Accelerator specification in 'TYPE' or 'TYPE:count' format.",  # Optional: Add description
        )
        # Remove properties/required if they were inferred from the object structure
        schema.pop("properties", None)
        schema.pop("required", None)
        return schema


class ModelRepo(custom_types.ConfigModel):
    repo_id: str
    revision: Optional[str] = None
    allow_patterns: Optional[list[str]] = None
    ignore_patterns: Optional[list[str]] = None

    @pydantic.model_validator(mode="after")
    def _check_repo(self) -> "ModelRepo":
        if not self.repo_id:
            raise ValidationError("Repo ID  for Hugging Face model cannot be empty")
        return self


class ModelCache(pydantic.RootModel[list[ModelRepo]]):
    @property
    def models(self) -> list[ModelRepo]:
        return self.root


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
    secret_to_path_mapping: dict[str, str] = pydantic.Field(default_factory=dict)

    _secret_name_regex: ClassVar[re.Pattern] = re.compile(r"^[-._a-zA-Z0-9]+$")

    @staticmethod
    def validate_secret_name(secret_name: str) -> None:
        if not isinstance(secret_name, str) or not secret_name:
            raise ValueError(f"Invalid secret name `{secret_name}`")

        if len(secret_name) > 253:
            raise ValueError(f"Secret name `{secret_name}` is too long.")

        if secret_name in {".", ".."}:
            raise ValueError(f"Secret name `{secret_name}` cannot be `{secret_name}`.")

        # Mimic k8s sanitization
        k8s_safe = re.sub("[^0-9a-zA-Z]+", "-", secret_name)
        if secret_name != k8s_safe and not secret_name.startswith(
            REGISTRY_BUILD_SECRET_PREFIX
        ):
            raise ValueError(
                f"Secrets used in builds must follow Kubernetes object naming conventions. Name `{secret_name}` is not valid. "
                f"Please use only alphanumeric characters and `-`."
            )

    @pydantic.model_validator(mode="after")
    def _validate_secrets(self) -> "Build":
        for secret_name, path in self.secret_to_path_mapping.items():
            self.validate_secret_name(secret_name)
        return self


class Resources(custom_types.ConfigModel):
    cpu: str = DEFAULT_CPU
    memory: str = DEFAULT_MEMORY
    use_gpu: bool = DEFAULT_USE_GPU
    accelerator: AcceleratorSpec = pydantic.Field(default_factory=AcceleratorSpec)
    node_count: Optional[Annotated[int, pydantic.Field(ge=1, strict=True)]] = None

    _milli_cpu_regex: ClassVar[re.Pattern] = re.compile(r"^[0-9.]*m$")
    _memory_regex: ClassVar[re.Pattern] = re.compile(r"^[0-9.]*([a-zA-Z]+)?$")
    _memory_units: ClassVar[dict[str, int]] = {
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

    @property
    def memory_in_bytes(self) -> int:
        match = self._memory_regex.search(self.memory)
        assert match
        unit = match.group(1)
        return math.ceil(float(self.memory.strip(unit)) * self._memory_units[unit])

    @pydantic.field_validator("accelerator", mode="before")
    def _default_accelerator_if_none(cls, v: Any) -> AcceleratorSpec:
        return AcceleratorSpec() if v is None else v

    @pydantic.field_validator("cpu")
    def validate_cpu(cls, v: str) -> str:
        try:
            float(v)
            return v
        except ValueError:
            if not cls._milli_cpu_regex.fullmatch(v):
                raise ValidationError(f"Invalid cpu specification {v}")
        return v

    @pydantic.field_validator("memory")
    def validate_memory(cls, v: str) -> str:
        try:
            float(v)
            return v
        except ValueError:
            match = cls._memory_regex.fullmatch(v)
            if not match:
                raise ValidationError(f"Invalid memory specification {v}")
            unit = match.group(1)
            if unit not in cls._memory_units:
                raise ValidationError(f"Invalid memory unit {unit} in {v}")
        return v

    @pydantic.model_validator(mode="after")
    def _sync_gpu_flag(self) -> "Resources":
        if self.accelerator.accelerator is not None:
            self.use_gpu = True
        return self

    @pydantic.model_serializer(mode="wrap")
    def _serialize(
        self,
        handler: core_schema.SerializerFunctionWrapHandler,
        info: core_schema.SerializationInfo,
    ) -> dict:
        result = handler(self)
        if not self.node_count:
            result.pop("node_count", None)
        return result


class ExternalDataItem(custom_types.ConfigModel):
    """A piece of remote data, to be made available to the Truss at serving time.

    Remote data is downloaded and stored under Truss's data directory. Care should be taken
    to avoid conflicts. This will get precedence if there's overlap.
    """

    # Url to download the data from.
    # Currently only files are allowed.
    url: str
    # This should be path relative to data directory. This is where the remote
    # file will be downloaded.
    local_data_path: str
    # This should be path relative to data directory. This is where the remote
    # file will be downloaded.
    backend: str = "http_public"
    # This should be path relative to data directory. This is where the remote
    # file will be downloaded.
    name: Optional[str] = None

    @pydantic.model_validator(mode="after")
    def _validate_paths(self) -> "ExternalDataItem":
        if not self.url:
            raise ValueError("URL of an external data item cannot be empty")
        if not self.local_data_path:
            raise ValueError(
                "The `local_data_path` field of an external data item cannot be empty"
            )
        return self


class ExternalData(custom_types.ConfigModel):
    """[Experimental] External data is data that is not contained in the Truss folder.

    Typically this will be data stored remotely. This data is guaranteed to be made
    available under the data directory of the truss."""

    items: list[ExternalDataItem]


class DockerAuthType(str, enum.Enum):
    """This enum will express all of the types of registry
    authentication we support."""

    GCP_SERVICE_ACCOUNT_JSON = "GCP_SERVICE_ACCOUNT_JSON"


class DockerAuthSettings(custom_types.ConfigModel):
    """Provides information about how to authenticate to the docker registry containing
    the custom base image."""

    auth_method: DockerAuthType
    secret_name: str
    registry: Optional[str] = ""

    @pydantic.field_validator("auth_method", mode="before")
    @classmethod
    def _normalize_auth_method(cls, v: str) -> str:
        return v.upper() if isinstance(v, str) else v


class BaseImage(custom_types.ConfigModel):
    image: str = ""
    python_executable_path: str = ""
    docker_auth: Optional[DockerAuthSettings] = None

    @pydantic.field_validator("python_executable_path")
    def _validate_path(cls, v: str) -> str:
        if v and not PurePosixPath(v).is_absolute():
            raise ValidationError(
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
    _MODEL_NAME_RE: ClassVar[re.Pattern] = re.compile(r"^[a-zA-Z0-9_-]+-[0-9a-f]{8}$")

    model_framework: str = "custom"
    model_type: str = "Model"
    model_name: Optional[str] = None
    model_module_dir: str = DEFAULT_MODEL_MODULE_DIR
    model_class_filename: str = "model.py"
    model_class_name: str = "Model"

    data_dir: str = DEFAULT_DATA_DIRECTORY
    external_data: Optional[ExternalData] = None

    input_type: str = "Any"
    model_metadata: dict[str, Any] = pydantic.Field(default_factory=dict)
    requirements_file: Optional[str] = None
    requirements: list[str] = pydantic.Field(default_factory=list)
    system_packages: list[str] = pydantic.Field(default_factory=list)
    environment_variables: dict[str, str] = pydantic.Field(default_factory=dict)
    resources: Resources = pydantic.Field(default_factory=Resources)
    runtime: Runtime = pydantic.Field(default_factory=Runtime)
    build: Build = pydantic.Field(default_factory=Build)
    python_version: str = "py39"
    examples_filename: str = "examples.yaml"
    secrets: dict[str, str] = pydantic.Field(default_factory=dict)
    description: Optional[str] = None
    bundled_packages_dir: str = DEFAULT_BUNDLED_PACKAGES_DIR
    external_package_dirs: list[str] = pydantic.Field(default_factory=list)
    base_image: Optional[BaseImage] = None
    docker_server: Optional[DockerServer] = None
    model_cache: ModelCache = pydantic.Field(default_factory=lambda: ModelCache([]))
    trt_llm: Optional[trt_llm_config.TRTLLMConfiguration] = None
    build_commands: list[str] = pydantic.Field(default_factory=list)

    # Internal
    use_local_src: bool = False
    cache_internal: CacheInternal = pydantic.Field(
        default_factory=lambda: CacheInternal([])
    )
    live_reload: bool = False
    apply_library_patches: bool = True
    spec_version: str = "2.0"

    def to_dict(self, verbose: bool = False) -> dict:
        data = super().to_dict(verbose)
        # Always include.
        data["resources"] = self.resources.to_dict(verbose=True)
        data["python_version"] = self.python_version
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "TrussConfig":
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
            if "hf_cache" in raw_data:
                logger.warning(
                    """Warning: `hf_cache` is deprecated in favor of `model_cache`.
                    Everything will run as before, but if you are pulling weights from S3 or GCS, they will be
                    stored at /app/model_cache instead of /app/hf_cache as before."""
                )
        return cls.from_dict(raw_data)

    def write_to_yaml_file(self, path: pathlib.Path, verbose: bool = True):
        with path.open("w") as config_file:
            yaml.safe_dump(self.to_dict(verbose=verbose), config_file)

    def clone(self) -> "TrussConfig":
        return self.from_dict(self.to_dict())

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

    @pydantic.field_validator("model_name")
    def validate_model_name(cls, model_name: str) -> str:
        if not model_name:
            return model_name

        if not bool(cls._MODEL_NAME_RE.match(model_name)):
            raise ValueError(
                f"Model name `{model_name}` must match regex {cls._MODEL_NAME_RE}."
            )
        return model_name

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
    def _default_cache_internal_if_none(cls, v: Any) -> AcceleratorSpec:
        return CacheInternal([]) if v is None else v  # type: ignore[return-value]

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


def _infer_python_version() -> str:
    return f"py{sys.version_info.major}{sys.version_info.minor}"


def map_local_to_supported_python_version() -> str:
    return _map_to_supported_python_version(_infer_python_version())


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
