import pathlib
from pathlib import Path
from typing import List, Optional

from truss.base import truss_config
from truss.base.custom_types import SafeModel
from truss.cli.utils.output import console
from truss.remote.baseten import custom_types as b10_types
from truss.remote.baseten.api import BasetenApi
from truss.remote.baseten.core import archive_dir
from truss.remote.baseten.remote import BasetenRemote
from truss.remote.baseten.utils import transfer
from truss_train.definitions import (
    InteractiveSession,
    InteractiveSessionTrigger,
    TrainingJob,
    TrainingProject,
)


class S3Artifact(SafeModel):
    s3_bucket: str
    s3_key: str


# Performs validation/transformation of fields that we don't want to expose
# to the end user via the TrainingJob SDK.
class PreparedTrainingJob(TrainingJob):
    runtime_artifacts: List[S3Artifact] = []
    truss_user_env: Optional[b10_types.TrussUserEnv] = None

    def model_dump(self, *args, **kwargs):
        data = super().model_dump(*args, **kwargs)

        # NB(nikhil): nest our processed artifacts back under runtime
        data["runtime"]["artifacts"] = data.pop("runtime_artifacts")
        return data


def prepare_push(
    api: BasetenApi,
    config: pathlib.Path,
    training_job: TrainingJob,
    truss_user_env: Optional[b10_types.TrussUserEnv] = None,
):
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
        interactive_session=training_job.interactive_session,
        runtime_artifacts=[
            S3Artifact(s3_key=credentials["s3_key"], s3_bucket=credentials["s3_bucket"])
        ],
        truss_user_env=truss_user_env,
    )


def _upsert_project_and_create_job(
    remote_provider: BasetenRemote,
    training_project: TrainingProject,
    config: Path,
    team_id: Optional[str] = None,
) -> dict:
    project_resp = remote_provider.upsert_training_project(
        training_project=training_project, team_id=team_id
    )

    # Collect TrussUserEnv with git info from the config directory
    working_dir = config.absolute().parent
    truss_user_env = b10_types.TrussUserEnv.collect_with_git_info(working_dir)

    prepared_job = prepare_push(
        remote_provider.api, config, training_project.job, truss_user_env=truss_user_env
    )

    job_resp = remote_provider.api.create_training_job(
        project_id=project_resp["id"], job=prepared_job
    )
    return job_resp


def create_training_job(
    remote_provider: BasetenRemote,
    config: Path,
    training_project: TrainingProject,
    job_name_from_cli: Optional[str] = None,
    team_name: Optional[str] = None,
    team_id: Optional[str] = None,
    interactive_trigger: Optional[str] = None,
    interactive_timeout_minutes: Optional[int] = None,
    accelerator: Optional[str] = None,
    node_count: Optional[int] = None,
    entrypoint: Optional[tuple[str, ...]] = None,
) -> dict:
    if job_name_from_cli:
        if training_project.job.name:
            console.print(
                f"[bold yellow]⚠ Warning:[/bold yellow] name '{training_project.job.name}' provided in config file will be ignored. Using job name '{job_name_from_cli}' provided via --job-name flag."
            )
        training_project.job.name = job_name_from_cli
    if team_name:
        training_project.team_name = team_name

    if interactive_trigger is not None or interactive_timeout_minutes is not None:
        if training_project.job.interactive_session is None:
            training_project.job.interactive_session = InteractiveSession()

        if interactive_trigger is not None:
            trigger_enum = InteractiveSessionTrigger(interactive_trigger.lower())
            existing_trigger = training_project.job.interactive_session.trigger
            if existing_trigger != InteractiveSessionTrigger.ON_DEMAND:
                console.print(
                    f"[bold yellow]⚠ Warning:[/bold yellow] interactive trigger '{existing_trigger.value}' provided in config file will be ignored. Using '{interactive_trigger}' provided via --interactive flag."
                )
            training_project.job.interactive_session.trigger = trigger_enum

        if interactive_timeout_minutes is not None:
            existing_timeout = training_project.job.interactive_session.timeout_minutes
            default_timeout = 8 * 60  # Default is 8 hours
            if existing_timeout != default_timeout:
                console.print(
                    f"[bold yellow]⚠ Warning:[/bold yellow] interactive timeout '{existing_timeout}' minutes provided in config file will be ignored. Using '{interactive_timeout_minutes}' minutes provided via --interactive-timeout-minutes flag."
                )
            training_project.job.interactive_session.timeout_minutes = (
                interactive_timeout_minutes
            )

    if accelerator is not None:
        existing_accelerator = training_project.job.compute.accelerator
        if existing_accelerator is not None:
            console.print(
                f"[bold yellow]⚠ Warning:[/bold yellow] accelerator '{existing_accelerator}' provided in config file will be ignored. Using '{accelerator}' provided via --accelerator flag."
            )
        training_project.job.compute.accelerator = truss_config.AcceleratorSpec(
            accelerator
        )

    if node_count is not None:
        existing_node_count = training_project.job.compute.node_count
        if existing_node_count != 1:  # 1 is the default
            console.print(
                f"[bold yellow]⚠ Warning:[/bold yellow] node_count '{existing_node_count}' provided in config file will be ignored. Using '{node_count}' provided via --node-count flag."
            )
        training_project.job.compute.node_count = node_count

    if entrypoint is not None:
        existing_entrypoint = training_project.job.runtime.entrypoint
        if existing_entrypoint:
            console.print(
                f"[bold yellow]⚠ Warning:[/bold yellow] entrypoint {existing_entrypoint} provided in config file will be ignored. Using entrypoint provided via --entrypoint flags."
            )
        training_project.job.runtime.entrypoint = list(entrypoint)

    job_resp = _upsert_project_and_create_job(
        remote_provider=remote_provider,
        training_project=training_project,
        config=config,
        team_id=team_id,
    )
    job_resp["job_object"] = training_project.job
    return job_resp
