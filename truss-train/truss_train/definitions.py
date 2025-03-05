from typing import Dict, List, Optional, Union

from pydantic import Field
from truss.base import truss_config
from truss.base.custom_types import SafeModel


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
    job: TrainingJob = Field(exclude=True)
