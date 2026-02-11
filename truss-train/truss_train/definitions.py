import enum
from abc import ABC
from typing import Dict, List, Literal, Optional, Union

import pydantic
from pydantic import ValidationError, field_validator, model_validator

from truss.base import constants, custom_types, truss_config

DEFAULT_LORA_RANK = 16
DEFAULT_INTERACTIVE_SESSION_TIMEOUT_MINUTES = 8 * 60

# Allowed LoRA rank values for vLLM
ALLOWED_LORA_RANKS = {8, 16, 32, 64, 128, 256, 320, 512}


class ModelWeightsFormat(str, enum.Enum):
    """Predefined supported model weights formats for deploying model from checkpoints via `truss train deploy_checkpoints`."""

    LORA = "lora"
    FULL = "full"
    WHISPER = "whisper"

    def to_truss_config(self) -> "ModelWeightsFormat":
        return ModelWeightsFormat[self.name]


class SecretReference(custom_types.SafeModelNoExtra):
    name: str


class Compute(custom_types.SafeModelNoExtra):
    node_count: int = 1
    cpu_count: int = 1
    memory: str = "2Gi"
    accelerator: Optional[truss_config.AcceleratorSpec] = None

    def model_dump(self, *args, **kwargs):
        data = super().model_dump(*args, **kwargs)
        if self.accelerator and self.accelerator.accelerator:
            data["accelerator"] = {
                "accelerator": self.accelerator.accelerator.value,
                "count": self.accelerator.count,
            }
        return data

    def to_truss_config(self) -> truss_config.Resources:
        if self.accelerator:
            return truss_config.Resources(
                cpu=str(self.cpu_count),
                memory=self.memory,
                accelerator=self.accelerator,
                node_count=self.node_count,
            )
        return truss_config.Resources(
            cpu=str(self.cpu_count), memory=self.memory, node_count=self.node_count
        )


class _CheckpointBase(custom_types.SafeModelNoExtra):
    typ: str


class _BasetenLatestCheckpoint(_CheckpointBase):
    job_id: Optional[str] = None
    project_name: Optional[str] = None
    typ: Literal["baseten_latest_checkpoint"] = "baseten_latest_checkpoint"


class _BasetenNamedCheckpoint(_CheckpointBase):
    checkpoint_name: str
    job_id: str
    typ: Literal["baseten_named_checkpoint"] = "baseten_named_checkpoint"


class BasetenCheckpoint:
    @staticmethod
    def from_latest_checkpoint(
        project_name: Optional[str] = None, job_id: Optional[str] = None
    ) -> _BasetenLatestCheckpoint:
        if not job_id and not project_name:
            raise ValidationError("job_id or project_name is required")
        return _BasetenLatestCheckpoint(project_name=project_name, job_id=job_id)

    @classmethod
    def from_named_checkpoint(
        cls, checkpoint_name: str, job_id: str
    ) -> _BasetenNamedCheckpoint:
        return _BasetenNamedCheckpoint(checkpoint_name=checkpoint_name, job_id=job_id)


class LoadCheckpointConfig(custom_types.SafeModelNoExtra):
    enabled: bool = False
    checkpoints: List[Union[_BasetenLatestCheckpoint, _BasetenNamedCheckpoint]] = [
        _BasetenLatestCheckpoint()
    ]
    download_folder: str = constants.DEFAULT_TRAINING_CHECKPOINT_FOLDER


class CheckpointingConfig(custom_types.SafeModelNoExtra):
    enabled: bool = False
    checkpoint_path: Optional[str] = None
    volume_size_gib: Optional[int] = None


class CacheConfig(custom_types.SafeModelNoExtra):
    enabled: bool = False
    enable_legacy_hf_mount: bool = False
    require_cache_affinity: bool = True
    mount_base_path: str = "/root/.cache"


class InteractiveSessionTrigger(str, enum.Enum):
    ON_STARTUP = "on_startup"
    ON_FAILURE = "on_failure"
    ON_DEMAND = "on_demand"


class InteractiveSessionProvider(str, enum.Enum):
    VS_CODE = "vs_code"
    CURSOR = "cursor"


class InteractiveSessionAuthProvider(str, enum.Enum):
    GITHUB = "github"
    MICROSOFT = "microsoft"


class InteractiveSession(custom_types.SafeModelNoExtra):
    trigger: InteractiveSessionTrigger = InteractiveSessionTrigger.ON_DEMAND
    timeout_minutes: int = DEFAULT_INTERACTIVE_SESSION_TIMEOUT_MINUTES
    session_provider: InteractiveSessionProvider = InteractiveSessionProvider.VS_CODE
    auth_provider: InteractiveSessionAuthProvider = (
        InteractiveSessionAuthProvider.GITHUB
    )


class Runtime(custom_types.SafeModelNoExtra):
    start_commands: List[str] = []
    environment_variables: Dict[str, Union[str, SecretReference]] = {}
    enable_cache: Optional[bool] = None
    checkpointing_config: CheckpointingConfig = CheckpointingConfig()
    load_checkpoint_config: Optional[LoadCheckpointConfig] = None
    cache_config: Optional[CacheConfig] = None

    @model_validator(mode="before")
    @classmethod
    def validate_cache_config(cls, values):
        enable_cache = values.get("enable_cache")
        cache_config = values.get("cache_config")

        if enable_cache is not None and cache_config is not None:
            raise ValueError(
                "Cannot set both 'enable_cache' and 'cache_config'. "
                "'enable_cache' is deprecated. Prefer migrating to 'cache_config' with "
                "`enabled=True` and `enable_legacy_hf_mount=True`."
            )

        # Migrate enable_cache to cache_config if enable_cache is True
        if enable_cache is not None and cache_config is None:
            values["cache_config"] = CacheConfig(
                enabled=enable_cache, enable_legacy_hf_mount=enable_cache
            )

        values.pop(
            "enable_cache", None
        )  # Remove deprecated field or else it will fail server-side validation

        return values


class AWSIAMDockerAuth(custom_types.SafeModelNoExtra):
    access_key_secret_ref: SecretReference
    secret_access_key_secret_ref: SecretReference


class GCPServiceAccountJSONDockerAuth(custom_types.SafeModelNoExtra):
    service_account_json_secret_ref: SecretReference


class DockerAuth(custom_types.SafeModelNoExtra):
    auth_method: truss_config.DockerAuthType
    registry: str
    aws_iam_docker_auth: Optional[AWSIAMDockerAuth] = None
    gcp_service_account_json_docker_auth: Optional[GCPServiceAccountJSONDockerAuth] = (
        None
    )


class Image(custom_types.SafeModelNoExtra):
    base_image: str
    docker_auth: Optional[DockerAuth] = None


class Workspace(custom_types.SafeModelNoExtra):
    workspace_root: Optional[str] = None
    external_dirs: List[str] = []
    exclude_dirs: List[str] = []


class TrainingJob(custom_types.SafeModelNoExtra):
    image: Image
    compute: Compute = Compute()
    runtime: Runtime = Runtime()
    interactive_session: Optional[InteractiveSession] = None
    name: Optional[str] = None
    workspace: Optional[Workspace] = None
    weights: List[truss_config.WeightsSource] = []
    """MDN weight sources to mount in the training container. Weights are mirrored and cached for fast startup."""

    def model_dump(self, *args, **kwargs):
        data = super().model_dump(*args, **kwargs)
        data["compute"] = self.compute.model_dump()
        return data


class TrainingProject(custom_types.SafeModelNoExtra):
    name: str
    # TrainingProject is the wrapper around project config and job config. However, we exclude job
    # in serialization so just TrainingProject metadata is included in API requests.
    job: TrainingJob = pydantic.Field(exclude=True)
    team_name: Optional[str] = None


class Checkpoint(custom_types.ConfigModel, ABC):
    training_job_id: str
    checkpoint_name: str
    model_weight_format: ModelWeightsFormat

    def to_truss_config(self) -> truss_config.TrainingArtifactReference:
        return truss_config.TrainingArtifactReference(
            training_job_id=self.training_job_id,
            paths=[f"rank-0/{self.checkpoint_name}/"],
        )


class LoRADetails(custom_types.ConfigModel):
    """Configuration details specific to LoRA (Low-Rank Adaptation) models."""

    rank: int = DEFAULT_LORA_RANK

    @field_validator("rank")
    @classmethod
    def validate_lora_rank(cls, v):
        if v not in ALLOWED_LORA_RANKS:
            raise ValueError(
                f"lora_rank ({v}) must be one of {sorted(ALLOWED_LORA_RANKS)}. Got {v}.model_weight_format = checkpoints[0].model_weight_format"
            )
        return v


class FullCheckpoint(Checkpoint):
    model_weight_format: ModelWeightsFormat = ModelWeightsFormat.FULL


class WhisperCheckpoint(Checkpoint):
    model_weight_format: ModelWeightsFormat = ModelWeightsFormat.WHISPER


class LoRACheckpoint(Checkpoint):
    lora_details: LoRADetails = LoRADetails()
    model_weight_format: ModelWeightsFormat = ModelWeightsFormat.LORA


class CheckpointList(custom_types.SafeModelNoExtra):
    download_folder: str = truss_config.DEFAULT_TRAINING_CHECKPOINT_FOLDER
    base_model_id: Optional[str] = None
    checkpoints: List[Checkpoint] = []

    def to_truss_config(self) -> truss_config.CheckpointList:
        artifact_references: List[truss_config.TrainingArtifactReference] = [
            checkpoint.to_truss_config() for checkpoint in self.checkpoints
        ]
        return truss_config.CheckpointList(
            download_folder=self.download_folder,
            artifact_references=artifact_references,
        )


class DeployCheckpointsRuntime(custom_types.SafeModelNoExtra):
    environment_variables: Dict[str, Union[str, SecretReference]] = {}


class DeployCheckpointsConfig(custom_types.SafeModelNoExtra):
    checkpoint_details: Optional[CheckpointList] = None
    model_name: Optional[str] = None
    runtime: Optional[DeployCheckpointsRuntime] = None
    compute: Optional[Compute] = None


class MemoryRequirements(custom_types.SafeModelNoExtra):
    """Memory scoping for auto-selecting H100, H200, or multinode H100."""

    model_params_b: Optional[float] = None
    """Model size in billions (e.g. 7 for 7B). Inferred from model name if not set."""

    per_device_batch_size: int = 2
    """Per-device batch size (affects activation memory)."""

    max_seq_length: int = 2048
    """Max sequence length (affects activation memory)."""


class AutoSFT(custom_types.SafeModelNoExtra):
    """Configuration for supervised fine-tuning (SFT) training.

    Load from a config.yaml or config.py file via `truss train sft config.yaml`.
    """

    model: str
    """Model identifier (e.g. HuggingFace model ID or path)."""

    dataset: str
    """Dataset identifier (e.g. HuggingFace dataset ID or path)."""

    num_epochs: int
    """Number of training epochs."""

    optimizer: Optional[str] = None
    """Optimizer name (e.g. 'adam', 'adamw', 'sgd')."""

    learning_rate: Optional[float] = None
    """Learning rate."""

    lr_scheduler: Optional[str] = None
    """Learning rate scheduler (e.g. 'cosine', 'linear', 'constant_with_warmup')."""

    # Dataset loading (UniversalLLMLoader options)
    max_samples: Optional[int] = None
    """Limit to first N samples; auto-enables streaming for HF Hub."""

    split: Optional[str] = None
    """Dataset split (e.g. 'train', 'validation')."""

    # Training project options
    project_name: Optional[str] = None
    """Training project name (default: derived from model)."""

    base_image: Optional[str] = None
    """Docker base image for training (default: pytorch with transformers)."""

    accelerator: Optional[str] = None
    """Accelerator spec. When memory or model_params_b is set, auto-scoped to H100, H200, or multinode H100."""

    node_count: Optional[int] = None
    """Number of compute nodes. Megatron (ms-swift) is auto-selected when > 1."""

    framework: Optional[str] = None
    """Training framework: 'transformers', 'trl', 'megatron'. Auto-selected if not specified (megatron when multinode)."""

    model_params_b: Optional[float] = None
    """Model size in billions (e.g. 235 for 235B). Drives hardware scoping when set. Overrides memory.model_params_b."""

    memory: Optional[MemoryRequirements] = None
    """Memory scoping. When set (or model_params_b is set), accelerator and node_count are auto-derived (H100/H200/multinode H100)."""

    environment_variables: Optional[Dict[str, Union[str, Dict[str, str]]]] = None
    """Extra env vars for the training container. Use {"secret": "name"} for secrets.
    HF_TOKEN with hf_access_token is included by default."""
