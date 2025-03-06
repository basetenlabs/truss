from typing import Dict, List, Optional, Union

import pydantic

from truss.base import truss_config


class SafeModel(pydantic.BaseModel):
    """Pydantic base model with reasonable config."""

    model_config = pydantic.ConfigDict(
        arbitrary_types_allowed=False, strict=True, validate_assignment=True
    )


class SecretReference(SafeModel):
    name: str


class Compute(SafeModel):
    node_count: int = 1
    cpu_count: int = 1
    memory: str = "2Gi"
    accelerator: Optional[truss_config.AcceleratorSpec] = None


class Runtime(SafeModel):
    start_commands: List[str] = []
    environment_variables: Dict[str, Union[str, SecretReference]] = {}


class Image(SafeModel):
    base_image: str


class TrainingJob(SafeModel):
    image: Image
    compute: Compute = Compute()
    runtime: Runtime = Runtime()


class TrainingProject(SafeModel):
    name: str
    # TrainingProject is the wrapper around project config and job config. However, we exclude job
    # in serialization so just TrainingProject metadata is included in API requests.
    job: TrainingJob = pydantic.Field(exclude=True)
