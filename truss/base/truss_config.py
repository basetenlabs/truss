import enum
import logging
import math
import os
import pathlib
import re
import sys
import warnings
from functools import cached_property
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

# PORT: knative reserved
# HOSTNAME: set to the pod name by k8s
K8S_RESERVED_ENVIRONMENT_VARIABLES = {"PORT", "HOSTNAME"}

from truss.base.constants import PYPROJECT_TOML_FILENAME, UV_LOCK_FILENAME
from truss.util.requirements import (
    parse_requirement_string,
    parse_requirements_from_pyproject,
    raise_insufficent_revision,
)
from truss.util.yaml_utils import safe_load_yaml_with_no_duplicates

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

WEIGHTS_AUTH_SECRET_NAME_PARAM = "auth_secret_name"
DOCKER_AUTH_SECRET_NAME_PARAM = "secret_name"
AWS_OIDC_ROLE_ARN_PARAM = "aws_oidc_role_arn"
AWS_OIDC_REGION_PARAM = "aws_oidc_region"
GCP_OIDC_SERVICE_ACCOUNT_PARAM = "gcp_oidc_service_account"
GCP_OIDC_WORKLOAD_ID_PROVIDER_PARAM = "gcp_oidc_workload_id_provider"


def _is_numeric(number_like: str) -> bool:
    try:
        float(number_like)
        return True
    except ValueError:
        return False


class RequirementsFileType(str, enum.Enum):
    NOT_PROVIDED = "not_provided"
    PIP = "pip"
    PYPROJECT = "pyproject"

    # NB(nikhil): `uv.lock` requires the sibling `pyproject.toml`, so we need to make some assumptions about
    # the location of that file.
    UV_LOCK = "uv_lock"


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
    L40S = "L40S"
    RTX_PRO_6000 = "RTX_PRO_6000"


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
    def __get_pydantic_json_schema__(
        cls,
        core_schema: pydantic_core.CoreSchema,
        handler: pydantic.GetJsonSchemaHandler,
    ) -> json_schema.JsonSchemaValue:
        schema = handler(core_schema)
        schema.update(type="string")
        schema.pop("properties", None)
        schema.pop("required", None)
        return schema


class ModelRepoSourceKind(str, enum.Enum):
    """syned with `pub enum ResolutionType` in truss-transfer"""

    HF = "hf"
    GCS = "gcs"
    S3 = "s3"
    AZURE = "azure"


class ModelRepo(custom_types.ConfigModel):
    repo_id: Annotated[str, pydantic.StringConstraints(min_length=1)]
    revision: str = ""
    allow_patterns: Optional[list[str]] = None
    ignore_patterns: Optional[list[str]] = None
    volume_folder: Optional[
        Annotated[str, pydantic.StringConstraints(min_length=1)]
    ] = None
    use_volume: bool
    kind: ModelRepoSourceKind = ModelRepoSourceKind.HF
    runtime_secret_name: str = "hf_access_token"

    @pydantic.field_validator("revision")
    @classmethod
    def _validate_revision(cls, v: str) -> str:
        if len(v) == 1:
            raise ValueError("revision must be empty or at least 2 characters")
        return v

    @property
    def runtime_path(self) -> pathlib.PurePosixPath:
        assert self.volume_folder is not None
        return constants.MODEL_CACHE_PATH / self.volume_folder

    @pydantic.model_validator(mode="before")
    def _check_v2_requirements(cls, v) -> str:
        use_volume = v.get("use_volume", False)
        if not use_volume:
            return v
        revision = v.get("revision") or ""
        kind = v.get("kind")
        is_hf = kind is None or kind == ModelRepoSourceKind.HF.value
        if is_hf and not revision:
            logger.warning(
                "the key `revision: str` is required for use_volume=True huggingface repos."
            )
            raise_insufficent_revision(v.get("repo_id"), revision)
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


class ModelRepoCacheInternal(ModelRepo):
    use_volume: bool = False  # override


class CacheInternal(pydantic.RootModel[list[ModelRepoCacheInternal]]):
    @property
    def models(self) -> list[ModelRepoCacheInternal]:
        return self.root


class WeightsAuthMethod(str, enum.Enum):
    """Authentication methods for weights sources."""

    CUSTOM_SECRET = "CUSTOM_SECRET"
    AWS_OIDC = "AWS_OIDC"
    GCP_OIDC = "GCP_OIDC"


class AuthFieldsMixin(custom_types.ConfigModel):
    """Mixin for common authentication fields used across different auth configurations."""

    aws_oidc_role_arn: Optional[str] = pydantic.Field(
        default=None, description="AWS IAM role ARN for OIDC authentication."
    )
    aws_oidc_region: Optional[str] = pydantic.Field(
        default=None, description="AWS region for OIDC authentication."
    )
    gcp_oidc_service_account: Optional[str] = pydantic.Field(
        default=None, description="GCP service account name for OIDC authentication."
    )
    gcp_oidc_workload_id_provider: Optional[str] = pydantic.Field(
        default=None,
        description="GCP workload identity provider for OIDC authentication.",
    )

    def _require_fields(self, auth_method: str, *fields: str) -> None:
        """Validate that all specified fields have non-empty values.

        Args:
            auth_method: The authentication method being validated (for error messages)
            fields: Field names to check for presence

        Raises:
            ValueError: If any required fields are missing or empty
        """
        missing = [f for f in fields if getattr(self, f) in (None, "")]
        if missing:
            raise ValueError(
                f"{', '.join(missing)} must be provided when auth_method is {auth_method}"
            )

    def _forbid_fields(self, auth_method: str, *fields: str) -> None:
        """Validate that all specified fields are empty or None.

        Args:
            auth_method: The authentication method being validated (for error messages)
            fields: Field names to check for absence

        Raises:
            ValueError: If any forbidden fields are present
        """
        present = [f for f in fields if getattr(self, f) not in (None, "")]
        if present:
            raise ValueError(
                f"{', '.join(present)} cannot be specified when auth_method is {auth_method}"
            )

    def _validate_fields(
        self, auth_method: str, required: list[str], forbidden: list[str]
    ) -> None:
        """Validate that required fields are present and forbidden fields are absent.

        Args:
            auth_method: The authentication method being validated (for error messages)
            required: List of field names that must have values
            forbidden: List of field names that must be empty/None
        """
        self._require_fields(auth_method, *required)
        self._forbid_fields(auth_method, *forbidden)


class WeightsAuth(AuthFieldsMixin):
    """Authentication configuration for a weights source.

    This can be used to specify OIDC-based authentication for cloud storage sources,
    or a Baseten secret name for access key authentication.
    """

    auth_method: Annotated[
        WeightsAuthMethod,
        pydantic.Field(
            ...,
            description="Authentication method for downloading weights from the source.",
        ),
    ]
    auth_secret_name: Optional[str] = pydantic.Field(
        default=None,
        description="Baseten secret name containing credentials for accessing the source.",
    )

    @pydantic.field_validator("auth_method", mode="before")
    @classmethod
    def _normalize_auth_method(cls, v: Optional[str]) -> Optional[str]:
        return v.upper() if isinstance(v, str) else v

    @pydantic.model_validator(mode="after")
    def _validate_auth_fields(self) -> "WeightsAuth":
        if self.auth_method == WeightsAuthMethod.CUSTOM_SECRET:
            self._validate_fields(
                self.auth_method.value,
                required=[WEIGHTS_AUTH_SECRET_NAME_PARAM],
                forbidden=[
                    AWS_OIDC_ROLE_ARN_PARAM,
                    AWS_OIDC_REGION_PARAM,
                    GCP_OIDC_SERVICE_ACCOUNT_PARAM,
                    GCP_OIDC_WORKLOAD_ID_PROVIDER_PARAM,
                ],
            )
        elif self.auth_method == WeightsAuthMethod.AWS_OIDC:
            self._validate_fields(
                self.auth_method.value,
                required=[AWS_OIDC_ROLE_ARN_PARAM, AWS_OIDC_REGION_PARAM],
                forbidden=[
                    WEIGHTS_AUTH_SECRET_NAME_PARAM,
                    GCP_OIDC_SERVICE_ACCOUNT_PARAM,
                    GCP_OIDC_WORKLOAD_ID_PROVIDER_PARAM,
                ],
            )
        elif self.auth_method == WeightsAuthMethod.GCP_OIDC:
            self._validate_fields(
                self.auth_method.value,
                required=[
                    GCP_OIDC_SERVICE_ACCOUNT_PARAM,
                    GCP_OIDC_WORKLOAD_ID_PROVIDER_PARAM,
                ],
                forbidden=[
                    WEIGHTS_AUTH_SECRET_NAME_PARAM,
                    AWS_OIDC_ROLE_ARN_PARAM,
                    AWS_OIDC_REGION_PARAM,
                ],
            )

        return self


# URI prefixes for cloud storage sources
_CLOUD_STORAGE_PREFIXES = frozenset({"s3://", "gs://", "azure://", "r2://"})
# HuggingFace prefix
_HF_PREFIX = "hf://"
# HTTPS prefix for direct URL downloads
_HTTPS_PREFIX = "https://"
# All supported URI schemes (cloud storage + HuggingFace + HTTPS)
_SUPPORTED_SCHEMES = _CLOUD_STORAGE_PREFIXES | {_HF_PREFIX, _HTTPS_PREFIX}


class WeightsSource(custom_types.ConfigModel):
    """Configuration for a weights source in the new weights API.

    Uses a URI-based `source` field with a required scheme prefix:
    - hf:// -> HuggingFace (e.g., "hf://meta-llama/Llama-2-7b" or "hf://meta-llama/Llama-2-7b@main")
    - s3:// -> AWS S3 (e.g., "s3://bucket/path")
    - gs:// -> Google Cloud Storage (e.g., "gs://bucket/path")
    - azure:// -> Azure Blob Storage (e.g., "azure://account/container/path")
    - r2:// -> CloudFlare R2 Storage (e.g., "r2://account_id.bucket/path")
    - https:// -> Direct URL download (e.g., "https://example.com/model.bin")

    For HuggingFace sources, you can specify a revision (branch, tag, or commit SHA)
    using the @{rev} suffix: "hf://owner/repo@revision"

    Authentication can be specified either:
    - Using the `auth` section (required for OIDC):
        auth:
          auth_method: AWS_OIDC
          aws_oidc_role_arn: <role_arn>
          aws_oidc_region: <region>
    - Using `auth_secret_name` at the top level (or in the `auth` section)
    """

    source: Annotated[str, pydantic.StringConstraints(min_length=1)] = pydantic.Field(
        ...,
        description="URI with scheme prefix. Use hf://, s3://, gs://, azure://, r2://, or https://. "
        "For HuggingFace, use @revision suffix (e.g., hf://owner/repo@main).",
    )
    mount_location: Annotated[str, pydantic.StringConstraints(min_length=1)] = (
        pydantic.Field(
            ..., description="Absolute path where weights will be mounted at runtime."
        )
    )
    auth: Optional[WeightsAuth] = pydantic.Field(
        default=None,
        description="Authentication configuration for accessing the weights source.",
    )
    auth_secret_name: Optional[str] = pydantic.Field(
        default=None,
        description="Baseten secret name containing credentials. Can also be specified in auth.auth_secret_name.",
    )
    allow_patterns: Optional[list[str]] = pydantic.Field(
        default=None, description="File patterns to include (e.g., ['*.safetensors'])."
    )
    ignore_patterns: Optional[list[str]] = pydantic.Field(
        default=None, description="File patterns to exclude (e.g., ['*.md'])."
    )

    @property
    def is_huggingface(self) -> bool:
        """Check if this source is a HuggingFace repository."""
        return self.source.startswith(_HF_PREFIX)

    @pydantic.field_validator("source")
    @classmethod
    def _validate_source(cls, v: str) -> str:
        supported_schemes_str = ", ".join(sorted(_SUPPORTED_SCHEMES))

        # URI scheme prefix is required
        if "://" not in v:
            raise ValueError(
                f"Source '{v}' is missing a URI scheme. "
                f"Supported schemes: {supported_schemes_str}"
            )

        scheme = v.split("://")[0] + "://"

        # Check for unsupported URI schemes
        if scheme not in _SUPPORTED_SCHEMES:
            raise ValueError(
                f"Unsupported source scheme '{scheme}'. "
                f"Supported schemes: {supported_schemes_str}"
            )

        # Validate URI format for cloud storage
        if scheme in _CLOUD_STORAGE_PREFIXES:
            path_part = v[len(scheme) :]
            if not path_part or path_part.startswith("/"):
                raise ValueError(
                    f"Invalid {scheme[:-3].upper()} URI format: '{v}'. "
                    f"Expected format: {scheme}bucket/path"
                )
            # Reject @ revision syntax for cloud storage (HF-only feature)
            if "@" in path_part:
                raise ValueError(
                    f"The @ revision syntax is only valid for HuggingFace sources (hf://). "
                    f"Source '{v}' uses {scheme[:-3].upper()} which does not support revisions."
                )

        # Validate https:// format
        if scheme == _HTTPS_PREFIX:
            url_part = v[len(_HTTPS_PREFIX) :]
            if not url_part or url_part.startswith("/"):
                raise ValueError(
                    f"Invalid HTTPS URL format: '{v}'. "
                    f"Expected format: https://hostname/path"
                )

        # Validate hf:// format
        if scheme == _HF_PREFIX:
            repo_part = v[len(_HF_PREFIX) :]
            if not repo_part or repo_part.startswith("/"):
                raise ValueError(
                    f"Invalid HuggingFace URI format: '{v}'. "
                    f"Expected format: hf://owner/repo"
                )

        return v

    @pydantic.field_validator("mount_location")
    @classmethod
    def _validate_mount_location(cls, v: str) -> str:
        if not v.startswith("/"):
            raise ValueError(
                f"mount_location must be an absolute path (start with /), got: {v}"
            )
        return v

    @pydantic.model_validator(mode="after")
    def _validate_auth_secret_name(self) -> "WeightsSource":
        """Validate that auth_secret_name is not specified in conflicting locations."""
        if self.auth_secret_name and (self.auth and self.auth.auth_secret_name):
            raise ValueError(
                "auth_secret_name cannot be specified both at the top level and in auth section. "
                "Please use only one location."
            )
        return self


class Weights(pydantic.RootModel[list[WeightsSource]]):
    """List of weights sources for the new weights API."""

    @property
    def sources(self) -> list[WeightsSource]:
        return self.root

    @pydantic.model_validator(mode="after")
    def _validate_unique_mount_locations(self) -> "Weights":
        """Ensure all mount_location values are unique."""
        mount_locations: list[str] = []
        for source in self.root:
            if source.mount_location in mount_locations:
                raise ValueError(
                    f"Duplicate mount_location '{source.mount_location}' - "
                    f"each weights source must have a unique mount path."
                )
            mount_locations.append(source.mount_location)
        return self


class HealthChecks(custom_types.ConfigModel):
    """Custom health check configuration for your deployments."""

    restart_check_delay_seconds: Optional[int] = pydantic.Field(
        default=None,
        description="The delay in seconds before starting restart checks. Defaults to platform-determined value when not set.",
    )
    restart_threshold_seconds: Optional[int] = pydantic.Field(
        default=None,
        description="The time in seconds after which an unhealthy instance is restarted. Defaults to platform-determined value when not set.",
    )
    stop_traffic_threshold_seconds: Optional[int] = pydantic.Field(
        default=None,
        description="The time in seconds after which traffic is stopped to an unhealthy instance. Defaults to platform-determined value when not set.",
    )
    startup_threshold_seconds: Optional[int] = pydantic.Field(
        default=None,
        description="The time in seconds to wait for a model to start before marking it as unhealthy. Defaults to platform-determined value when not set.",
    )


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


class RemoteSSH(custom_types.ConfigModel):
    """Configuration for SSH access to running model instances."""

    enabled: bool = pydantic.Field(
        default=False,
        description="If true, enables SSH access to running model instances.",
    )


class Runtime(custom_types.ConfigModel):
    """Runtime settings for your model instance."""

    predict_concurrency: int = pydantic.Field(
        default=1,
        description="The number of concurrent requests that can run in your model's predict method. Increase this if your model supports parallelism.",
    )
    streaming_read_timeout: int = pydantic.Field(
        default=60, description="The timeout in seconds for streaming read operations."
    )
    enable_tracing_data: bool = pydantic.Field(
        default=False,
        description="If true, enables trace data export with built-in OTEL instrumentation. May add performance overhead.",
    )
    enable_debug_logs: bool = pydantic.Field(
        default=False,
        description="If true, sets the Truss server log level to DEBUG instead of INFO.",
    )
    transport: Transport = pydantic.Field(
        default_factory=HTTPOptions,
        description="The transport protocol for your model. Supports http (default), websocket, and grpc.",
    )
    is_websocket_endpoint: Optional[bool] = pydantic.Field(
        None,
        description="DEPRECATED. Do not set manually. Automatically inferred from transport.kind == websocket.",
    )
    health_checks: HealthChecks = pydantic.Field(
        default_factory=HealthChecks,
        description="Custom health check configuration for your deployments.",
    )
    remote_ssh: RemoteSSH = pydantic.Field(
        default_factory=RemoteSSH,
        description="Configuration for SSH access to running model instances.",
    )
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
    """Build-time configuration, including secret access during Docker builds."""

    model_server: ModelServer = ModelServer.TrussServer
    arguments: dict[str, Any] = pydantic.Field(default_factory=dict)
    secret_to_path_mapping: Mapping[str, str] = pydantic.Field(
        default_factory=dict,
        description="Grants access to secrets during the build. Provide a mapping between a secret and a path on the image.",
    )
    no_cache: bool = False

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
    """Compute resources that your model needs, including CPU, memory, and GPU resources."""

    cpu: str = pydantic.Field(
        default=DEFAULT_CPU,
        description="CPU resources needed, expressed as either a raw number or millicpus. For example, 500m is half of a CPU core.",
        examples=["1", "500m", "4"],
    )
    memory: str = pydantic.Field(
        default=DEFAULT_MEMORY,
        description="CPU RAM needed, expressed as a number with units. Units include Gi (Gibibytes), G (Gigabytes), Mi (Mebibytes), and M (Megabytes).",
        examples=["2Gi", "512Mi"],
    )
    accelerator: AcceleratorSpec = pydantic.Field(
        default_factory=AcceleratorSpec,
        description="The GPU type for your instance. To request multiple GPUs, use the ':' operator (e.g. L4:4).",
        examples=["A100", "T4:2", "H100:8"],
    )
    instance_type: Optional[str] = pydantic.Field(
        default=None,
        description="The full SKU name for the instance type. When specified, cpu, memory, and accelerator fields are ignored.",
        examples=["L4:4x16"],
    )
    node_count: Optional[Annotated[int, pydantic.Field(ge=1, strict=True)]] = (
        pydantic.Field(
            default=None, description="Number of nodes for multi-node deployments."
        )
    )

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
        """Custom omission of `node_count` and `instance_type` if at default."""
        result = handler(self)
        if not self.node_count:
            result.pop("node_count", None)
        if not self.instance_type:
            result.pop("instance_type", None)
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
    AWS_OIDC = "AWS_OIDC"
    GCP_OIDC = "GCP_OIDC"
    REGISTRY_SECRET = "REGISTRY_SECRET"


class DockerAuthSettings(AuthFieldsMixin):
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
    def validate_auth_fields(self) -> "DockerAuthSettings":
        if self.auth_method == DockerAuthType.GCP_SERVICE_ACCOUNT_JSON:
            self._validate_fields(
                self.auth_method.value,
                required=[DOCKER_AUTH_SECRET_NAME_PARAM],
                forbidden=[
                    AWS_OIDC_ROLE_ARN_PARAM,
                    AWS_OIDC_REGION_PARAM,
                    GCP_OIDC_SERVICE_ACCOUNT_PARAM,
                    GCP_OIDC_WORKLOAD_ID_PROVIDER_PARAM,
                ],
            )
        elif self.auth_method == DockerAuthType.AWS_OIDC:
            self._validate_fields(
                self.auth_method.value,
                required=[AWS_OIDC_ROLE_ARN_PARAM, AWS_OIDC_REGION_PARAM],
                forbidden=[
                    DOCKER_AUTH_SECRET_NAME_PARAM,
                    GCP_OIDC_SERVICE_ACCOUNT_PARAM,
                    GCP_OIDC_WORKLOAD_ID_PROVIDER_PARAM,
                ],
            )
        elif self.auth_method == DockerAuthType.GCP_OIDC:
            self._validate_fields(
                self.auth_method.value,
                required=[
                    GCP_OIDC_SERVICE_ACCOUNT_PARAM,
                    GCP_OIDC_WORKLOAD_ID_PROVIDER_PARAM,
                ],
                forbidden=[
                    DOCKER_AUTH_SECRET_NAME_PARAM,
                    AWS_OIDC_ROLE_ARN_PARAM,
                    AWS_OIDC_REGION_PARAM,
                ],
            )

        return self


class BaseImage(custom_types.ConfigModel):
    """Use base_image to deploy a custom Docker image."""

    image: str = pydantic.Field(
        default="",
        description="The path to the Docker image.",
        examples=["vllm/vllm-openai:v0.7.3", "nvcr.io/nvidia/nemo:23.03"],
    )
    python_executable_path: str = pydantic.Field(
        default="",
        description="A path to the Python executable on the image.",
        examples=["/usr/bin/python"],
    )
    docker_auth: Optional[DockerAuthSettings] = pydantic.Field(
        default=None,
        description="Authentication configuration for a private Docker registry.",
    )

    @pydantic.field_validator("python_executable_path")
    def _validate_path(cls, v: str) -> str:
        if v and not pathlib.PurePosixPath(v).is_absolute():
            raise ValueError(
                f"Invalid relative python executable path {v}. Provide an absolute path"
            )
        return v


class DockerServer(custom_types.ConfigModel):
    """Deploy a custom Docker image that has its own HTTP server, without writing a Model class."""

    start_command: Optional[str] = pydantic.Field(
        default=None,
        description="The command to start the server. Required when no_build is not true.",
    )
    server_port: int = pydantic.Field(
        description="The port where the server runs. Port 8080 is reserved by Baseten's internal reverse proxy and cannot be used."
    )
    predict_endpoint: str = pydantic.Field(
        description="The endpoint for inference requests. This is mapped to Baseten's /predict route."
    )
    readiness_endpoint: str = pydantic.Field(
        description="The endpoint for readiness probes. Determines when the container can accept traffic."
    )
    liveness_endpoint: str = pydantic.Field(
        description="The endpoint for liveness probes. Determines if the container needs to be restarted."
    )
    run_as_user_id: Optional[int] = pydantic.Field(
        default=None,
        description="The Linux UID to run the server process as inside the container. Use this when your base image expects a specific non-root user (for example, NVIDIA NIM containers).",
    )
    no_build: Optional[bool] = pydantic.Field(
        default=None,
        description="Skip the build step and deploy the base image as-is. Baseten copies the image to its container registry without running docker build or modifying the image in any way.",
    )

    @pydantic.field_validator("run_as_user_id")
    @classmethod
    def _validate_run_as_user_id(cls, v: Optional[int]) -> Optional[int]:
        if v == 0 or v == constants.DEFAULT_NON_ROOT_USER_ID:
            raise ValueError(f"run_as_user_id cannot be {v}. Use a different user ID.")
        return v

    @pydantic.model_validator(mode="after")
    def _validate_start_command(self) -> "DockerServer":
        if not self.no_build and self.start_command is None:
            raise ValueError("start_command is required when no_build is not true")
        return self


class TrainingArtifactReference(custom_types.ConfigModel):
    training_job_id: str = pydantic.Field(
        ..., description="The training job id that the artifact reference belongs to."
    )
    paths: list[str] = pydantic.Field(
        default_factory=list,
        description="The paths of the files to download which can contain * or ?.",
    )


class CheckpointList(custom_types.ConfigModel):
    download_folder: str = pydantic.Field(
        default=DEFAULT_TRAINING_CHECKPOINT_FOLDER,
        description="The folder to download the checkpoints to.",
        examples=["/tmp/training_checkpoints"],
    )
    artifact_references: list[TrainingArtifactReference] = pydantic.Field(
        default_factory=list
    )


# TODO: remove just use normal python version instead of this.
def to_dotted_python_version(truss_python_version: str) -> str:
    """Converts python version string using in truss config to the conventional dotted form.

    e.g. py39 to 3.9
    """
    return f"{truss_python_version[2]}.{truss_python_version[3:]}"


class TrussConfig(custom_types.ConfigModel):
    """Configuration for a Truss model deployment."""

    model_name: Optional[str] = pydantic.Field(
        default=None,
        description="The name of your model. This is displayed in the model details page in the Baseten UI.",
    )
    model_metadata: dict[str, Any] = pydantic.Field(
        default_factory=dict,
        description="A flexible field for additional metadata. The entire config file is available to your model at runtime.",
        json_schema_extra={
            "properties": {
                "example_model_input": {
                    "description": "Sample input that populates the Baseten playground.",
                    "examples": [{"prompt": "What is the meaning of life?"}],
                }
            }
        },
    )
    description: Optional[str] = pydantic.Field(
        default=None, description="A description of your model."
    )
    examples_filename: str = pydantic.Field(
        default="examples.yaml",
        description="Path to a file containing example model inputs.",
    )

    data_dir: str = pydantic.Field(
        default=DEFAULT_DATA_DIRECTORY,
        description="The folder for data files in your Truss.",
    )
    external_data: Optional[ExternalData] = pydantic.Field(
        default=None,
        description="External data to be downloaded and made available under the data directory at serving time.",
    )
    external_package_dirs: list[str] = pydantic.Field(
        default_factory=list,
        description="Use external_package_dirs to access custom packages located outside your Truss. This lets multiple Trusses share the same package.",
    )

    python_version: str = pydantic.Field(
        default="py313",
        description="The Python version to use.",
        examples=["py313", "py312", "py311", "py310", "py39"],
    )
    base_image: Optional[BaseImage] = pydantic.Field(
        default=None,
        description="Use a custom Docker base image instead of the default Truss image.",
    )
    requirements_file: Optional[str] = pydantic.Field(
        default=None,
        description="Path to a dependency file. Supports requirements.txt, pyproject.toml, and uv.lock. Mutually exclusive with 'requirements'.",
    )
    requirements: list[str] = pydantic.Field(
        default_factory=list,
        description="A list of Python dependencies in pip requirements file format. Mutually exclusive with 'requirements_file'.",
    )
    system_packages: list[str] = pydantic.Field(
        default_factory=list,
        description="System packages that you would typically install using apt on a Debian operating system.",
        examples=[["ffmpeg", "libsm6", "libxext6"]],
    )
    environment_variables: dict[str, str] = pydantic.Field(
        default_factory=dict,
        description="Key-value pairs exposed to the environment that the model executes in. Do not store secret values here.",
    )
    secrets: MutableMapping[str, Optional[str]] = pydantic.Field(
        default_factory=dict,
        description="Declare secrets your model needs at runtime, such as API keys or access tokens. Use null as a placeholder; store actual values in your organization settings.",
    )

    resources: Resources = pydantic.Field(
        default_factory=Resources,
        description="Compute resources that your model needs, including CPU, memory, and GPU resources.",
    )
    runtime: Runtime = pydantic.Field(
        default_factory=Runtime, description="Runtime settings for your model instance."
    )
    build: Build = pydantic.Field(
        default_factory=Build,
        description="Build-time configuration, including secret access during Docker builds.",
    )
    build_commands: list[str] = pydantic.Field(
        default_factory=list,
        description="A list of shell commands to run during Docker build. These commands execute after system packages and Python requirements are installed.",
    )
    docker_server: Optional[DockerServer] = pydantic.Field(
        default=None,
        description="Deploy a custom Docker image that has its own HTTP server, without writing a Model class.",
    )
    model_cache: ModelCache = pydantic.Field(
        default_factory=lambda: ModelCache([]),
        description="Deprecated. Use 'weights' instead. Bundle model weights into your image at build time.",
    )
    weights: Weights = pydantic.Field(
        default_factory=lambda: Weights([]),
        description="Configure Baseten Delivery Network (BDN) for model weight delivery with multi-tier caching.",
    )
    trt_llm: Optional[trt_llm_config.TRTLLMConfiguration] = pydantic.Field(
        default=None,
        description="TensorRT-LLM configuration for optimized LLM inference.",
    )

    # deploying from checkpoint
    training_checkpoints: Optional[CheckpointList] = pydantic.Field(
        default=None,
        description="Configuration for deploying from training checkpoints.",
    )

    # Internal / Legacy.
    input_type: str = "Any"
    model_framework: str = "custom"
    model_type: str = "Model"
    model_module_dir: str = pydantic.Field(
        default=DEFAULT_MODEL_MODULE_DIR,
        description="The folder containing your model class.",
    )
    model_class_filename: str = "model.py"
    model_class_name: str = pydantic.Field(
        default="Model",
        description="The name of the class that defines your Truss model. This class must implement at least a predict method.",
    )
    bundled_packages_dir: str = pydantic.Field(
        default=DEFAULT_BUNDLED_PACKAGES_DIR,
        description="The folder for custom packages in your Truss.",
    )
    use_local_src: bool = False
    cache_internal: CacheInternal = pydantic.Field(
        default_factory=lambda: CacheInternal([])
    )
    live_reload: bool = pydantic.Field(
        default=False,
        description="If true, changes to your model code are automatically reloaded without restarting the server.",
    )
    apply_library_patches: bool = pydantic.Field(
        default=True,
        description="Whether to apply library patches for improved compatibility.",
    )
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
        env_vars = data.get("environment_variables", {})
        conflicts = K8S_RESERVED_ENVIRONMENT_VARIABLES & env_vars.keys()
        if conflicts:
            logger.warning(
                "Warning: the following environment variables are reserved by the "
                "platform and will be overwritten at runtime: %s",
                ", ".join(sorted(conflicts)),
            )
        data["environment_variables"] = {
            k: str(v).lower() if isinstance(v, bool) else str(v)
            for k, v in env_vars.items()
        }
        return cls.model_validate(data)

    @classmethod
    def from_yaml(cls, path: pathlib.Path) -> "TrussConfig":
        if not os.path.isfile(path):
            # It's common for users to create a .yml instead of a .yaml,
            # so check for that and provide a helpful error message if we find one.
            resolved_path = path.resolve()
            stem = resolved_path.stem
            alternative_path = resolved_path.parent / f"{stem}.yml"
            if os.path.isfile(alternative_path):
                raise ValueError(
                    "No truss configuration file ending in .yaml but found one ending in .yml. Did you mean to rename it?"
                )
            else:
                raise ValueError(f"Expected a truss configuration file at {path}")

        with path.open() as f:
            raw_data = safe_load_yaml_with_no_duplicates(f) or {}
        # TODO(deepakn): Remove this once we have a way to pass no_cache through the context.
        build_section = raw_data.get("build")
        if isinstance(build_section, dict) and build_section.get("no_cache") is True:
            raise ValueError(
                "no_cache cannot be specified in config.yaml. Use the --no-cache CLI flag instead."
            )
        return cls.from_dict(raw_data)

    def write_to_yaml_file(self, path: pathlib.Path, verbose: bool = True):
        with path.open("w") as config_file:
            yaml.safe_dump(self.to_dict(verbose=verbose), config_file)

    def clone(self) -> "TrussConfig":
        return self.from_dict(self.to_dict())

    @cached_property
    def requirements_file_type(self) -> RequirementsFileType:
        return self._detect_requirements_file_type()

    def load_requirements_from_file(self, truss_dir: pathlib.Path) -> list[str]:
        file_type = self.requirements_file_type
        if file_type == RequirementsFileType.NOT_PROVIDED:
            return []

        try:
            if file_type == RequirementsFileType.PIP:
                return self._load_pip_requirements(truss_dir)

            # NB(nikhil): For patching, we resolve from `pyproject.toml` for (1) easier parsing (2) smaller file footprint.
            # If the user specified `uv.lock` as the source of truth, we'll bypass it for the patch process.
            pyproject_path = self._resolve_pyproject_path(truss_dir)
            return parse_requirements_from_pyproject(pyproject_path)
        except Exception as e:
            logger.exception(
                f"failed to read requirements file: {self.requirements_file}"
            )
            raise e

    def _load_pip_requirements(self, truss_dir: pathlib.Path) -> list[str]:
        requirements_path = truss_dir / self.requirements_file  # type: ignore[operator]
        requirements = []
        with open(requirements_path) as f:
            for line in f.readlines():
                parsed_line = parse_requirement_string(line)
                if parsed_line:
                    requirements.append(parsed_line)
        return requirements

    def _resolve_pyproject_path(self, truss_dir: pathlib.Path) -> pathlib.Path:
        if self.requirements_file_type == RequirementsFileType.PYPROJECT:
            return truss_dir / self.requirements_file  # type: ignore[operator]

        return (truss_dir / self.requirements_file).parent / PYPROJECT_TOML_FILENAME  # type: ignore[operator]

    @staticmethod
    def load_requirements_file_from_filepath(yaml_path: pathlib.Path) -> list[str]:
        config = TrussConfig.from_yaml(yaml_path)
        return config.load_requirements_from_file(yaml_path.parent)

    @pydantic.field_validator("python_version")
    def _validate_python_version(cls, v: str) -> str:
        valid = {f"py{x.replace('.', '')}" for x in constants.SUPPORTED_PYTHON_VERSIONS}
        if v not in valid:
            raise ValueError(f"Please ensure that `python_version` is one of {valid}")
        if v == "py39":
            warnings.warn(
                "Python 3.9 is deprecated and will be removed in a future release. "
                "Please upgrade to a newer Python version.",
                FutureWarning,
                stacklevel=2,
            )
        return v

    @pydantic.model_validator(mode="after")
    def _validate_remote_ssh(self) -> "TrussConfig":
        if (
            self.runtime.remote_ssh.enabled
            and self.docker_server is not None
            and self.docker_server.run_as_user_id is not None
        ):
            raise ValueError(
                "remote_ssh.enabled is not compatible with "
                "docker_server.run_as_user_id. SSH requires the default "
                "'app' user (uid 60000)."
            )
        return self

    @pydantic.model_validator(mode="after")
    def _validate_config(self) -> "TrussConfig":
        if self.requirements and self.requirements_file:
            raise ValueError(
                "Please ensure that only one of `requirements` and `requirements_file` is specified"
            )
        if self.model_cache.models and self.weights.sources:
            raise ValueError(
                "Please ensure that only one of `model_cache` and `weights` is specified"
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

    # NB(nikhil): clear_runtime_fields will remove all runtime specific fields from the config so
    # we can more optimally detect whether a new image build is needed.
    def clear_runtime_fields(self) -> None:
        self.training_checkpoints = None
        self.environment_variables = {}
        self.weights = Weights([])

    def _detect_requirements_file_type(self) -> RequirementsFileType:
        if not self.requirements_file:
            return RequirementsFileType.NOT_PROVIDED

        basename = pathlib.Path(self.requirements_file).name
        if basename == UV_LOCK_FILENAME:
            return RequirementsFileType.UV_LOCK
        elif basename == PYPROJECT_TOML_FILENAME:
            return RequirementsFileType.PYPROJECT
        return RequirementsFileType.PIP


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
