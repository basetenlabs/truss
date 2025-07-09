from typing import Dict, List, Optional, Union

import pydantic
from pydantic import field_validator, model_validator

from truss.base import custom_types, truss_config

DEFAULT_LORA_RANK = 16

# Allowed LoRA rank values for vLLM
ALLOWED_LORA_RANKS = {8, 16, 32, 64, 128, 256, 320, 512}


class SecretReference(custom_types.SafeModel):
    name: str


class Compute(custom_types.SafeModel):
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


class CheckpointingConfig(custom_types.SafeModel):
    enabled: bool = False
    checkpoint_path: Optional[str] = None


class CacheConfig(custom_types.SafeModel):
    enabled: bool = False
    enable_legacy_hf_mount: bool = False


class Runtime(custom_types.SafeModel):
    start_commands: List[str] = []
    environment_variables: Dict[str, Union[str, SecretReference]] = {}
    enable_cache: Optional[bool] = None
    checkpointing_config: CheckpointingConfig = CheckpointingConfig()
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


class Image(custom_types.SafeModel):
    base_image: str


class TrainingJob(custom_types.SafeModel):
    image: Image
    compute: Compute = Compute()
    runtime: Runtime = Runtime()

    def model_dump(self, *args, **kwargs):
        data = super().model_dump(*args, **kwargs)
        data["compute"] = self.compute.model_dump()
        return data


class TrainingProject(custom_types.SafeModel):
    name: str
    # TrainingProject is the wrapper around project config and job config. However, we exclude job
    # in serialization so just TrainingProject metadata is included in API requests.
    job: TrainingJob = pydantic.Field(exclude=True)


class Checkpoint(custom_types.SafeModel):
    training_job_id: str
    id: str
    name: str
    lora_rank: Optional[int] = (
        None  # lora rank will be fetched through the API if available.
    )

    @field_validator("lora_rank")
    @classmethod
    def validate_lora_rank(cls, v):
        if v is not None and v not in ALLOWED_LORA_RANKS:
            raise ValueError(
                f"lora_rank ({v}) must be one of {sorted(ALLOWED_LORA_RANKS)}"
            )
        return v

    def to_truss_config(self) -> truss_config.Checkpoint:
        return truss_config.Checkpoint(
            id=f"{self.training_job_id}/{self.id}", name=self.id
        )


class CheckpointList(custom_types.SafeModel):
    download_folder: str = truss_config.DEFAULT_TRAINING_CHECKPOINT_FOLDER
    base_model_id: Optional[str] = None
    checkpoints: List[Checkpoint] = []

    def to_truss_config(self) -> truss_config.CheckpointList:
        checkpoints = [checkpoint.to_truss_config() for checkpoint in self.checkpoints]
        return truss_config.CheckpointList(
            checkpoints=checkpoints, download_folder=self.download_folder
        )


class DeployCheckpointsRuntime(custom_types.SafeModel):
    environment_variables: Dict[str, Union[str, SecretReference]] = {}


class DeployCheckpointsConfig(custom_types.SafeModel):
    checkpoint_details: Optional[CheckpointList] = None
    model_name: Optional[str] = None
    deployment_name: Optional[str] = None
    runtime: Optional[DeployCheckpointsRuntime] = None
    compute: Optional[Compute] = None
