import base64
import pathlib

from truss.remote.baseten.api import BasetenApi
from truss.truss_train.definitions import SecretReference, TrainingJobSpec


def build_create_training_job_request(
    training_config_dir: pathlib.Path,
    api: BasetenApi,
    training_job_spec: TrainingJobSpec,
) -> dict:
    # serialize the file bundle for the framework_config_file
    framework_config_details = None
    if training_job_spec.training_config.framework_config_file:
        training_framework_config_path = pathlib.Path(
            training_job_spec.training_config.framework_config_file.source_path
        )
        if not training_framework_config_path.is_absolute():
            training_framework_config_path = (
                training_config_dir / training_framework_config_path
            )
        if not training_framework_config_path.exists():
            raise FileNotFoundError(
                f"Training framework config file not found: {training_framework_config_path}"
            )
        with open(training_framework_config_path, "rb") as f:
            config_contents = f.read()
        framework_config_details = {
            "content": base64.b64encode(config_contents).decode("utf-8"),
            "remote_path": training_job_spec.training_config.framework_config_file.remote_path,
        }

    request = {
        "hardware_config": {
            "instance_type": training_job_spec.hardware_config.dict(),
            "cloud_backed_volume": training_job_spec.hardware_config.cloud_backed_volume.dict(),
        },
        "runtime_config": {
            "image_name": training_job_spec.runtime_config.image_name,
            "environment_variables": {
                k: v.dict() if isinstance(v, SecretReference) else v
                for k, v in training_job_spec.runtime_config.environment_variables.items()
            },
            "start_commands": training_job_spec.runtime_config.start_commands,
        },
        "training_config": {
            "name": training_job_spec.training_config.name,
            # TODO: should we be uploading this to S3 and fetching it later?
            "framework_config_file": framework_config_details,
            "cloud_backed_volume_checkpoint_directory": training_job_spec.training_config.cloud_backed_volume_checkpoint_directory
            or training_job_spec.hardware_config.cloud_backed_volume.remote_mount_path,
        },
    }
    return request
