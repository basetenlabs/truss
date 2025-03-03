# Training types here
import pathlib
from typing import Dict, List, Optional, Union

from truss.base import truss_config
from truss.base.pydantic_models import SafeModel, SafeModelNonSerializable

LocalPath = Union[str, pathlib.Path]


class FileBundle(SafeModelNonSerializable):
    """A bundle of files to be copied into the docker image."""

    source_path: LocalPath
    remote_path: str


class SecretReference(SafeModel):
    name: str

    def dict(self, **kwargs):
        return {"name": self.name, "type": "secret"}


class InstanceType(truss_config.ComputeSpec):
    """Parsed and validated instance type."""

    ephemeral_storage: str = "512Gi"

    def dict(self, **kwargs):
        return {"ephemeral_storage": self.ephemeral_storage, **super().dict(**kwargs)}


class BasetenOutputStorage(SafeModel):
    remote_mount_path: str

    def dict(self, **kwargs):
        return {"remote_mount_path": self.remote_mount_path}


class HardwareConfig(SafeModel):
    instance_type: InstanceType
    cloud_backed_volume: Optional[BasetenOutputStorage] = None

    def dict(self, **kwargs):
        return {
            "instance_type": self.instance_type.dict(),
            "cloud_backed_volume": self.cloud_backed_volume.dict()
            if self.cloud_backed_volume
            else None,
        }


class TrainingConfig(SafeModel):
    name: str
    # relative to the BasetenOutputStorage "remote_mount_path"
    cloud_backed_volume_checkpoint_directory: Optional[str] = None


class RuntimeConfig(SafeModel):
    image_name: str
    start_commands: List[str]
    environment_variables: Dict[str, Union[str, SecretReference]] = {}
    file_bundles: List[FileBundle] = []


class TrainingJobSpec(SafeModel):
    hardware_config: HardwareConfig
    runtime_config: RuntimeConfig
    training_config: TrainingConfig
