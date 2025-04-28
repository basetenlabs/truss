import os
import tempfile
from dataclasses import dataclass
from typing import List, Optional, Tuple, cast

import rich
import rich_click as click
from InquirerPy import inquirer
from jinja2 import Template
from rich.console import Console

from truss.base.truss_config import AcceleratorSpec
from truss.cli.train.metrics_watcher import MetricsWatcher
from truss.remote.baseten.remote import BasetenRemote

ACTIVE_JOB_STATUSES = [
    "TRAINING_JOB_RUNNING",
    "TRAINING_JOB_CREATED",
    "TRAINING_JOB_DEPLOYING",
]

TRUSS_TEMPLATE = Template("""
base_image:
  image:  vllm/vllm-openai:latest
model_name: {{ model_name }}
training_checkpoints:
  download_folder: /tmp/training_checkpoints
  checkpoints:
  - id: {{ checkpoint_id }}
    name: {{ checkpoint_id }}
docker_server:
  start_command: sh -c "HF_TOKEN=$(cat /secrets/{{ hf_secret_name }}) vllm serve {{ base_model_id }} --port 8000 --tensor-parallel-size 4 --enable-lora --max-lora-rank 16 --dtype {{ dtype }} --lora-modules {{ checkpoint_id }}=/tmp/training_checkpoints/{{ checkpoint_id }}"
  readiness_endpoint: /health
  liveness_endpoint: /health
  predict_endpoint: /v1/chat/completions
  server_port: 8000
resources:
  accelerator: {{ accelerator }}
  use_gpu: true
runtime:
  predict_concurrency : 256
secrets:
  hf_access_token: set token in baseten workspace
environment_variables:
  VLLM_LOGGING_LEVEL: WARNING
  VLLM_USE_V1: 0
  HF_HUB_ENABLE_HF_TRANSFER: 1
""")


def get_args_for_stop(
    console: Console,
    remote_provider: BasetenRemote,
    project_id: Optional[str],
    job_id: Optional[str],
) -> Tuple[str, str]:
    if not project_id or not job_id:
        # get all running jobs
        job = _get_active_job(console, remote_provider, project_id, job_id)
        project_id_for_job = job["training_project"]["id"]
        job_id_to_stop = job["id"]
        # check if the user wants to stop the inferred running job
        if not job_id:
            confirm = inquirer.confirm(
                message=f"Are you sure you want to stop training job {job_id_to_stop}?",
                default=False,
            ).execute()
            if not confirm:
                raise click.UsageError("Training job not stopped.")
        return project_id_for_job, job_id_to_stop

    return project_id, job_id


def get_args_for_monitoring(
    console: Console,
    remote_provider: BasetenRemote,
    project_id: Optional[str],
    job_id: Optional[str],
) -> Tuple[str, str]:
    if not project_id or not job_id:
        jobs = remote_provider.api.search_training_jobs(
            project_id=project_id, job_id=job_id
        )
        if not jobs:
            raise click.UsageError("No jobs found.")
        if len(jobs) > 1:
            sorted_jobs = sorted(jobs, key=lambda x: x["created_at"], reverse=True)
            job = sorted_jobs[0]
            console.print(
                f"Multiple jobs found. Showing the most recently created job: {job['id']}",
                style="yellow",
            )
        else:
            job = jobs[0]
        project_id = cast(str, job["training_project"]["id"])
        job_id = cast(str, job["id"])
    return project_id, job_id


def _get_active_job(
    console: Console,
    remote_provider: BasetenRemote,
    project_id: Optional[str],
    job_id: Optional[str],
) -> dict:
    jobs = remote_provider.api.search_training_jobs(
        statuses=ACTIVE_JOB_STATUSES, project_id=project_id, job_id=job_id
    )
    if not jobs:
        raise click.UsageError("No running jobs found.")
    if len(jobs) > 1:
        display_training_jobs(console, jobs, title="Active Training Jobs")
        raise click.UsageError("Multiple active jobs found. Please specify a job id.")
    return jobs[0]


def display_training_jobs(console: Console, jobs, title="Training Job Details"):
    table = rich.table.Table(
        show_header=True,
        header_style="bold magenta",
        title=title,
        box=rich.table.box.ROUNDED,
        border_style="blue",
    )
    table.add_column("Project ID", style="cyan")
    table.add_column("Project Name", style="cyan")
    table.add_column("Job ID", style="cyan")
    table.add_column("Status", style="white")
    table.add_column("Instance Type", style="white")
    table.add_column("Created At", style="white")
    table.add_column("Updated At", style="white")
    for job in jobs:
        table.add_row(
            job["training_project"]["id"],
            job["training_project"]["name"],
            job["id"],
            job["current_status"],
            job["instance_type"]["name"],
            job["created_at"],
            job["updated_at"],
        )
    console.print(table)


def display_training_projects(console: Console, projects):
    table = rich.table.Table(
        show_header=True,
        header_style="bold magenta",
        title="Training Projects",
        box=rich.table.box.ROUNDED,
        border_style="blue",
    )
    table.add_column("Project ID", style="cyan")
    table.add_column("Name")
    table.add_column("Created At")
    table.add_column("Updated At")
    table.add_column("Latest Job ID")

    # most recent projects at bottom of terminal
    for project in projects[::-1]:
        latest_job = project.get("latest_job") or {}
        table.add_row(
            project["id"],
            project["name"],
            project["created_at"],
            project["updated_at"],
            latest_job.get("id", ""),
        )

    console.print(table)


def view_training_details(
    console: Console,
    remote_provider: BasetenRemote,
    project_id: Optional[str],
    job_id: Optional[str],
):
    """
    view_training_details shows a list of jobs that meet the provided project_id and job_id filters.

     If no filters are provided, the command will show a list of all training projects and a list of active jobs.
    """
    if job_id or project_id:
        jobs_response = remote_provider.api.search_training_jobs(
            project_id=project_id,
            job_id=job_id,
            order_by=[{"field": "created_at", "order": "asc"}],
        )
        if len(jobs_response) == 0:
            raise click.UsageError("No training jobs found")
        display_training_jobs(console, jobs_response)
    else:
        projects = remote_provider.api.list_training_projects()
        display_training_projects(console, projects)
        active_jobs = remote_provider.api.search_training_jobs(
            statuses=ACTIVE_JOB_STATUSES
        )
        if active_jobs:
            display_training_jobs(console, active_jobs, title="Active Training Jobs")
        else:
            console.print("No active training jobs.", style="yellow")


def stop_all_jobs(
    console: Console, remote_provider: BasetenRemote, project_id: Optional[str]
):
    active_jobs = remote_provider.api.search_training_jobs(
        project_id=project_id, statuses=ACTIVE_JOB_STATUSES
    )
    if not active_jobs:
        console.print("No active jobs found.", style="yellow")
        return
    confirm = inquirer.confirm(
        message=f"Are you sure you want to stop {len(active_jobs)} active jobs?",
        default=False,
    ).execute()
    if not confirm:
        raise click.UsageError("Training jobs not stopped.")
    for job in active_jobs:
        remote_provider.api.stop_training_job(job["training_project"]["id"], job["id"])
    console.print("Training jobs stopped successfully.", style="green")


def view_training_job_metrics(
    console: Console,
    remote_provider: BasetenRemote,
    project_id: Optional[str],
    job_id: Optional[str],
):
    """
    view_training_job_metrics shows a list of metrics for a training job.
    """
    project_id, job_id = get_args_for_monitoring(
        console, remote_provider, project_id, job_id
    )
    metrics_display = MetricsWatcher(remote_provider.api, project_id, job_id, console)
    metrics_display.watch()


@dataclass
class DeployCheckpointArgs:
    model_name: Optional[str]
    base_model_id: Optional[str]
    project_id: Optional[str]
    job_id: Optional[str]
    checkpoint_id: Optional[str]
    hf_secret_name: Optional[str]
    accelerator: Optional[str]
    dtype: Optional[str]


@dataclass
class PreparedCheckpointDeploy:
    truss_directory: str
    checkpoint_id: str
    model_name: str


def prepare_checkpoint_deploy(
    console: Console, remote_provider: BasetenRemote, args: DeployCheckpointArgs
) -> PreparedCheckpointDeploy:
    project_id, job_id = get_args_for_monitoring(
        console, remote_provider, args.project_id, args.job_id
    )
    deploy_args = {
        "base_model_id": args.base_model_id,
        "checkpoint_id": args.checkpoint_id,
        "hf_secret_name": args.hf_secret_name,
        "dtype": args.dtype,
    }
    deploy_args["accelerator"] = get_accelerator_str(args.accelerator)
    if not args.base_model_id:
        # prompt user for base model id
        base_model_id = inquirer.text(message="Enter the base model id.").execute()
        if not base_model_id:
            raise click.UsageError("Base model id is required.")
        deploy_args["base_model_id"] = base_model_id

    # get all checkpoints for the training job
    # response = remote_provider.api.list_training_job_checkpoints(project_id, job_id)
    # response_checkpoints = response["checkpoints"]
    response_checkpoints = [{"id": "checkpoint-1"}, {"id": "checkpoint-2"}]
    checkpoint_ids = [checkpoint["id"] for checkpoint in response_checkpoints]
    checkpoint_id = get_checkoint_id(args.checkpoint_id, checkpoint_ids)
    deploy_args["checkpoint_id"] = checkpoint_id
    model_name = (
        args.model_name
        or f"{deploy_args['base_model_id']}-{deploy_args['checkpoint_id']}"
    )
    deploy_args["model_name"] = model_name
    deploy_args["dtype"] = args.dtype or "bfloat16"
    deploy_args["hf_secret_name"] = get_hf_secret_name(console, args.hf_secret_name)
    # generate the truss config for vllm
    truss_config = TRUSS_TEMPLATE.render(**deploy_args)
    truss_directory = tempfile.mkdtemp(
        suffix=f"training-job-{job_id}-{deploy_args['checkpoint_id']}"
    )
    truss_config_path = os.path.join(truss_directory, "config.yaml")
    with open(truss_config_path, "w") as f:
        f.write(truss_config)
    console.print(truss_config, style="green")
    console.print(f"Writing truss config to {truss_directory}", style="yellow")
    return PreparedCheckpointDeploy(
        truss_directory=truss_directory,
        checkpoint_id=checkpoint_id,
        model_name=model_name,
    )


def get_checkoint_id(user_input: Optional[str], checkpoint_ids: List[str]) -> str:
    if user_input == "latest":
        return checkpoint_ids[-1]
    elif user_input:
        if user_input not in checkpoint_ids:
            raise click.UsageError(f"Invalid checkpoint id. Choices: {checkpoint_ids}")
        return user_input
    elif not user_input:
        if len(checkpoint_ids) == 0:
            raise click.UsageError("No checkpoints found for the training job.")
        if len(checkpoint_ids) > 1:
            checkpoint_id = inquirer.select(
                message="Select the checkpoint to deploy.", choices=checkpoint_ids
            ).execute()
            if not checkpoint_id:
                raise click.UsageError("Checkpoint id is required.")
        return checkpoint_id


def get_hf_secret_name(console: Console, user_input: Optional[str]) -> str:
    if not user_input:
        # prompt user for hf secret name
        hf_secret_name = inquirer.select(
            message="Enter the huggingface secret name.",
            choices=["hf_access_token", None, "custom"],
            default="hf_access_token",
        ).execute()
        if hf_secret_name == "custom":
            hf_secret_name = inquirer.text(
                message="Enter the huggingface secret name."
            ).execute()
        if not hf_secret_name:
            console.print("No hf secret name.", style="yellow")
        return hf_secret_name
    return user_input


def get_accelerator_str(user_input: Optional[str]) -> str:
    if not user_input:
        # prompt user for accelerator
        raw_accelerator = inquirer.text(
            message="Enter the accelerator to use for deployment."
        ).execute()
    accelerator_str = user_input or raw_accelerator
    # use this as validation
    AcceleratorSpec._from_string_spec(accelerator_str)
    return accelerator_str
