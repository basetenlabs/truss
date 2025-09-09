import pathlib
from pathlib import Path
from typing import List, Optional

from truss.base.custom_types import SafeModel
from truss.cli.utils.output import console
from truss.remote.baseten import custom_types as b10_types
from truss.remote.baseten.api import BasetenApi
from truss.remote.baseten.core import archive_dir
from truss.remote.baseten.remote import BasetenRemote
from truss.remote.baseten.utils import transfer
from truss_train import loader
from truss_train.definitions import TrainingJob, TrainingProject


class S3Artifact(SafeModel):
    s3_bucket: str
    s3_key: str


# Performs validation/transformation of fields that we don't want to expose
# to the end user via the TrainingJob SDK.
class PreparedTrainingJob(TrainingJob):
    runtime_artifacts: List[S3Artifact] = []

    def model_dump(self, *args, **kwargs):
        data = super().model_dump(*args, **kwargs)

        # NB(nikhil): nest our processed artifacts back under runtime
        data["runtime"]["artifacts"] = data.pop("runtime_artifacts")
        return data


def prepare_push(api: BasetenApi, config: pathlib.Path, training_job: TrainingJob):
    # Assume config is at the root of the directory.
    archive = archive_dir(config.absolute().parent)
    credentials = api.get_blob_credentials(b10_types.BlobType.TRAIN)
    transfer.multipart_upload_boto3(
        archive.name,
        credentials["s3_bucket"],
        credentials["s3_key"],
        credentials["creds"],
    )
    return PreparedTrainingJob(
        image=training_job.image,
        runtime=training_job.runtime,
        compute=training_job.compute,
        name=training_job.name,
        runtime_artifacts=[
            S3Artifact(s3_key=credentials["s3_key"], s3_bucket=credentials["s3_bucket"])
        ],
    )


def create_training_job(
    remote_provider: BasetenRemote, training_project: TrainingProject, config: Path
) -> dict:
    project_resp = remote_provider.api.upsert_training_project(
        training_project=training_project
    )
    prepared_job = prepare_push(remote_provider.api, config, training_project.job)

    job_resp = remote_provider.api.create_training_job(
        project_id=project_resp["id"], job=prepared_job
    )
    return job_resp


def create_training_job_from_file(
    remote_provider: BasetenRemote,
    config: Path,
    job_name_from_cli: Optional[str] = None,
) -> dict:
    with loader.import_training_project(config) as training_project:
        if job_name_from_cli:
            if training_project.job.name:
                console.print(
                    f"[bold yellow]⚠ Warning:[/bold yellow] name '{training_project.job.name}' provided in config file will be ignored. Using job name '{job_name_from_cli}' provided via --job-name flag."
                )
            training_project.job.name = job_name_from_cli
        job_resp = create_training_job(
            remote_provider=remote_provider,
            training_project=training_project,
            config=config,
        )
        job_resp["job_object"] = training_project.job
    return job_resp
