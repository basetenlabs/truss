from typing import Dict, List, Optional, Union

import pydantic

from truss.base import truss_config
from truss.shared import types


class SecretReference(types.SafeModel):
    name: str


class Compute(types.SafeModel):
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


class Runtime(types.SafeModel):
    start_commands: List[str] = []
    environment_variables: Dict[str, Union[str, SecretReference]] = {}
    use_read_write_cache: bool = False


class Image(types.SafeModel):
    base_image: str


class TrainingJob(types.SafeModel):
    image: Image
    compute: Compute = Compute()
    runtime: Runtime = Runtime()

    def model_dump(self, *args, **kwargs):
        data = super().model_dump(*args, **kwargs)
        data["compute"] = self.compute.model_dump()
        return data


class TrainingProject(types.SafeModel):
    name: str
    # TrainingProject is the wrapper around project config and job config. However, we exclude job
    # in serialization so just TrainingProject metadata is included in API requests.
    job: TrainingJob = pydantic.Field(exclude=True)
