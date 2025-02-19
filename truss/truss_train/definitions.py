# Training types here
from truss.base import truss_config
from truss.base.pydantic_models import SafeModel


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
    training_configuration: str  # File Bundle
    # relative to the BasetenOutputStorage "remote_mount_path"
    cloud_backed_checkpoint_directory: str


class TrainingJobSpec(SafeModel):
    hardware_config: HardwareConfig
    training_script: str
