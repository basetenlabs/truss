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


class BasetenOutputStorage(SafeModel):
    remote_mount_path: str


class HardwareConfig(SafeModel):
    instance_type: InstanceType
    cloud_backed_volume: BasetenOutputStorage


class TrainingConfig(SafeModel):
    name: str
    framework_config_file: Optional[FileBundle] = None
    # relative to the BasetenOutputStorage "remote_mount_path"
    cloud_backed_volume_checkpoint_directory: Optional[str] = None


class RuntimeConfig(SafeModel):
    image_name: str
    environment_variables: Dict[str, Union[str, SecretReference]]
    start_commands: List[str]


class TrainingJobSpec(SafeModel):
    hardware_config: HardwareConfig
    runtime_config: RuntimeConfig
    training_config: TrainingConfig
