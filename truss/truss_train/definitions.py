# Training types here
import pathlib
from typing import List, Union

from truss.base import truss_config
from truss.base.pydantic_models import SafeModel, SafeModelNonSerializable

LocalPath = Union[str, pathlib.Path]


class FileBundle(SafeModelNonSerializable):
    """A bundle of files to be copied into the docker image."""

    source_path: LocalPath
    remote_path: str


class InstanceType(SafeModel):
    """Parsed and validated instance type."""

    cpu_count: int = 1
    predict_concurrency: int = 1
    memory: str = "2Gi"
    accelerator: truss_config.AcceleratorSpec = truss_config.AcceleratorSpec()
    ephemeral_storage: str = "512Gi"


class BasetenOutputStorage(SafeModel):
    remote_mount_path: str


class HardwareConfig(SafeModel):
    instance_type: InstanceType
    cloud_backed_volume: BasetenOutputStorage


class TrainingConfig(SafeModel):
    name: str
    training_configuration: FileBundle
    # relative to the BasetenOutputStorage "remote_mount_path"
    cloud_backed_volume_checkpoint_directory: str


class RuntimeConfig(SafeModel):
    image_name: str
    environment_variables: str  # Envvar
    start_commands: List[str]


class TrainingJobSpec(SafeModel):
    hardware_config: HardwareConfig
    runtime_config: RuntimeConfig
    training_config: TrainingConfig
