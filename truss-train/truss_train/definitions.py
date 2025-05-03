from typing import Dict, List, Optional, Union

import pydantic

from truss.base import custom_types, truss_config

DEFAULT_LORA_RANK = 16


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


class CheckpointingConfig(custom_types.SafeModel):
    enabled: bool = False
    checkpoint_path: Optional[str] = None


class Runtime(custom_types.SafeModel):
    start_commands: List[str] = []
    environment_variables: Dict[str, Union[str, SecretReference]] = {}
    enable_cache: bool = False
    checkpointing_config: CheckpointingConfig = CheckpointingConfig()


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


class CheckpointDetails(custom_types.SafeModel):
    download_folder: str = truss_config.DEFAULT_TRAINING_CHECKPOINT_FOLDER
    base_model_id: Optional[str] = None
    checkpoints: List[Checkpoint] = []


class CheckpointDeployRuntime(custom_types.SafeModel):
    environment_variables: Dict[str, Union[str, SecretReference]] = {}


class CheckpointDeployConfig(custom_types.SafeModel):
    checkpoint_details: Optional[CheckpointDetails] = None
    model_name: Optional[str] = None
    base_model_id: Optional[str] = None
    deployment_name: Optional[str] = None
    runtime: Optional[CheckpointDeployRuntime] = None
    compute: Optional[Compute] = None
