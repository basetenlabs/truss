import enum
from abc import ABC
from typing import Dict, List, Literal, Optional, Union

import pydantic
from pydantic import ValidationError, field_validator, model_validator

from truss.base import constants, custom_types, truss_config

DEFAULT_LORA_RANK = 16

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


class CacheConfig(custom_types.SafeModelNoExtra):
    enabled: bool = False
    enable_legacy_hf_mount: bool = False
    require_cache_affinity: bool = True


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
                "`enabled=True` and `enable_legacy_hf_cache=True`."
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


class TrainingJob(custom_types.SafeModelNoExtra):
    image: Image
    compute: Compute = Compute()
    runtime: Runtime = Runtime()
    name: Optional[str] = None

    def model_dump(self, *args, **kwargs):
        data = super().model_dump(*args, **kwargs)
        data["compute"] = self.compute.model_dump()
        return data


class TrainingProject(custom_types.SafeModelNoExtra):
    name: str
    # TrainingProject is the wrapper around project config and job config. However, we exclude job
    # in serialization so just TrainingProject metadata is included in API requests.
    job: TrainingJob = pydantic.Field(exclude=True)


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
