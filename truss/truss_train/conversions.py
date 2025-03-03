import pathlib

from truss.remote.baseten.api import BasetenApi
from truss.remote.baseten.core import create_tar_with_progress_bar
from truss.remote.baseten.utils.transfer import multipart_upload_boto3
from truss.truss_train.definitions import SecretReference, TrainingJobSpec
from truss.util.path import handle_path_or_str


def build_create_training_job_request(
    training_config_dir: pathlib.Path,
    api: BasetenApi,
    training_job_spec: TrainingJobSpec,
    training_project_id: str,
) -> dict:
    hardware_config = training_job_spec.hardware_config.dict()
    # remove predict_concurrency from the instance type, as it is not expected by the jobs definition
    if hardware_config.get("instance_type", {}).get("predict_concurrency", None):
        hardware_config["instance_type"].pop("predict_concurrency")

    file_bundles = []
    subpath = f"training_project/{training_project_id}/blobs"
    for fb in training_job_spec.runtime_config.file_bundles:
        source_path = handle_path_or_str(fb.source_path)
        if not source_path.is_absolute():
            source_path = training_config_dir / source_path
        temp_file = create_tar_with_progress_bar(source_path)
        temp_credentials_s3_upload = api.create_blob_credentials(subpath)
        s3_key = temp_credentials_s3_upload.pop("s3_key")
        s3_bucket = temp_credentials_s3_upload.pop("s3_bucket")
        multipart_upload_boto3(
            temp_file.name, s3_bucket, s3_key, temp_credentials_s3_upload["creds"], None
        )

        file_bundles.append(
            {
                # We convert to a Path to validate, and back to a string for JSON serialization
                "remote_path": str(pathlib.Path(fb.remote_path)),
                "s3_key": s3_key,
            }
        )
    request = {
        "hardware_config": training_job_spec.hardware_config.dict(),
        "runtime_config": {
            "image_name": training_job_spec.runtime_config.image_name,
            "environment_variables": {
                k: v.dict() if isinstance(v, SecretReference) else v
                for k, v in training_job_spec.runtime_config.environment_variables.items()
            },
            "start_commands": training_job_spec.runtime_config.start_commands,
            "file_bundles": file_bundles,
        },
        "training_config": {"name": training_job_spec.training_config.name},
    }
    return request
