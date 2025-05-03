import os
import tempfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, cast

import rich
import rich_click as click
from InquirerPy import inquirer
from jinja2 import Template
from rich.console import Console
from rich.text import Text

from truss.base import truss_config
from truss.cli.train.metrics_watcher import MetricsWatcher
from truss.remote.baseten.remote import BasetenRemote
from truss_train import loader
from truss_train.definitions import (
    DEFAULT_LORA_RANK,
    Checkpoint,
    CheckpointDeploy,
    CheckpointDeployRuntime,
    CheckpointDetails,
    Compute,
    SecretReference,
)

ACTIVE_JOB_STATUSES = [
    "TRAINING_JOB_RUNNING",
    "TRAINING_JOB_CREATED",
    "TRAINING_JOB_DEPLOYING",
]


@dataclass
class PrepareCheckpointArgs:
    project_id: Optional[str]
    job_id: Optional[str]
    deploy_config_path: Optional[str]


@dataclass
class PrepareCheckpointResult:
    truss_directory: Path
    checkpoint_deploy: CheckpointDeploy


VLLM_LORA_START_COMMAND = Template(
    'sh -c "{%if envvars %}{{ envvars }} {% endif %}vllm serve {{ base_model_id }} --port 8000 --tensor-parallel-size 4 --enable-lora --max-lora-rank {{ max_lora_rank }} --dtype bfloat16 --lora-modules {{ lora_modules }}"'
)

HF_TOKEN_ENVVAR_NAME = "HF_TOKEN"


@dataclass
class DeployCheckpointTemplatingArgs:
    training_job_id: str
    checkpoint_deploy: CheckpointDeploy


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


def prepare_checkpoint_deploy(
    console: Console, remote_provider: BasetenRemote, args: PrepareCheckpointArgs
) -> PrepareCheckpointResult:
    project_id, job_id = get_args_for_monitoring(
        console, remote_provider, args.project_id, args.job_id
    )
    if not args.deploy_config_path:
        return _prepare_checkpoint_deploy(
            console, remote_provider, CheckpointDeploy(), project_id, job_id
        )
    #### User provided a checkpoint deploy config file
    with loader.import_target(
        args.deploy_config_path, CheckpointDeploy
    ) as checkpoint_deploy:
        return _prepare_checkpoint_deploy(
            console, remote_provider, checkpoint_deploy, project_id, job_id
        )


def _prepare_checkpoint_deploy(
    console: Console,
    remote_provider: BasetenRemote,
    deploy_config: CheckpointDeploy,
    project_id: str,
    job_id: str,
) -> PrepareCheckpointResult:
    response = remote_provider.api.list_training_job_checkpoints(project_id, job_id)
    response_checkpoints = OrderedDict(
        (checkpoint["checkpoint_id"], checkpoint)
        for checkpoint in response["checkpoints"]
    )
    checkpoint_details = deploy_config.checkpoint_details
    if not checkpoint_details:
        checkpoint_details = CheckpointDetails()
    # first, gather all checkpoint ids the user wants to deploy
    if not checkpoint_details.checkpoints:
        # allow the user to select a checkpoint
        checkpoint_ids = get_checkpoint_ids_to_deploy(list(response_checkpoints.keys()))
        checkpoint_details.checkpoints = [
            hydrate_checkpoints(checkpoint_id, response_checkpoints)
            for checkpoint_id in checkpoint_ids
        ]
        checkpoint_details.base_model_id = get_base_model_id(
            checkpoint_details.base_model_id, response_checkpoints[checkpoint_ids[0]]
        )
        deploy_config.checkpoint_details = checkpoint_details
    if not deploy_config.compute:
        deploy_config.compute = Compute()
    deploy_config.compute.accelerator = get_accelerator_if_specified(
        deploy_config.compute.accelerator
    )
    if not deploy_config.compute.accelerator:
        # default to CPU for local testing
        deploy_config.compute.node_count = 1
        deploy_config.compute.cpu_count = 1
        deploy_config.compute.memory = "0Mi"
    deploy_config.model_name = (
        deploy_config.model_name
        or f"{deploy_config.checkpoint_details.base_model_id.split('/')[-1]}-vLLM-LORA"  # current scope for deploying from checkpoint
    )
    if not deploy_config.runtime:
        deploy_config.runtime = CheckpointDeployRuntime()
    if not deploy_config.runtime.environment_variables:
        # Prompt the user for the huggingface secret name as a default. There's much more we could
        # do here, but we're keeping it simple for now.
        hf_secret_name = get_hf_secret_name(
            console,
            deploy_config.runtime.environment_variables.get(HF_TOKEN_ENVVAR_NAME),
        )
        deploy_config.runtime.environment_variables[HF_TOKEN_ENVVAR_NAME] = (
            SecretReference(name=hf_secret_name)
        )
    if not deploy_config.deployment_name:
        # use the first checkpoint id as the deployment name
        deploy_config.deployment_name = deploy_config.checkpoint_details.checkpoints[
            0
        ].id

    template_args = DeployCheckpointTemplatingArgs(
        training_job_id=job_id, checkpoint_deploy=deploy_config
    )

    rendered_truss = render_vllm_lora_truss_config(template_args)
    truss_directory = Path(tempfile.mkdtemp(suffix=f"training-job-{job_id}"))
    truss_config_path = truss_directory / "config.yaml"
    rendered_truss.write_to_yaml_file(truss_config_path)
    console.print(rendered_truss, style="green")
    console.print(f"Writing truss config to {truss_config_path}", style="yellow")
    return PrepareCheckpointResult(
        truss_directory=truss_directory, checkpoint_deploy=deploy_config
    )


def render_vllm_lora_truss_config(
    args: DeployCheckpointTemplatingArgs,
) -> truss_config.TrussConfig:
    deploy_config = truss_config.TrussConfig.from_yaml(
        Path(os.path.dirname(__file__), "deploy_from_checkpoint_config.yml")
    )
    if not deploy_config.docker_server:
        raise ValueError(
            "Unexpected checkpoint deployment config: missing docker_server"
        )
    if not deploy_config.training_checkpoints:
        raise ValueError(
            "Unexpected checkpoint deployment config: missing training_checkpoints"
        )

    for checkpoint in args.checkpoint_deploy.checkpoint_details.checkpoints:
        fully_qualified_checkpoint_id = f"{args.training_job_id}/{checkpoint.id}"
        deploy_config.training_checkpoints.checkpoints.append(
            truss_config.Checkpoint(
                id=fully_qualified_checkpoint_id, name=checkpoint.id
            )
        )
    deploy_config.model_name = args.checkpoint_deploy.model_name
    deploy_config.resources.accelerator = args.checkpoint_deploy.compute.accelerator
    deploy_config.resources.cpu = str(args.checkpoint_deploy.compute.cpu_count)
    deploy_config.resources.memory = args.checkpoint_deploy.compute.memory
    deploy_config.resources.node_count = args.checkpoint_deploy.compute.node_count
    for key, value in args.checkpoint_deploy.runtime.environment_variables.items():
        if isinstance(value, SecretReference):
            deploy_config.secrets[value.name] = "set token in baseten workspace"
        else:
            deploy_config.environment_variables[key] = value

    start_command_envvars = ""
    for key, value in args.checkpoint_deploy.runtime.environment_variables.items():
        # this is a quirk of serving vllm with secrets - we need to export the secret by cat-ing it
        if isinstance(value, SecretReference):
            deploy_config.secrets[value.name] = "set token in baseten workspace"
            start_command_envvars = f"{key}=$(cat /secrets/{value.name})"

    checkpoint_parts = []
    for checkpoint in args.checkpoint_deploy.checkpoint_details.checkpoints:
        ckpt_path = Path(
            args.checkpoint_deploy.checkpoint_details.download_directory, checkpoint.id
        )
        checkpoint_parts.append(f"{checkpoint.name}={ckpt_path}")
    checkpoint_str = " ".join(checkpoint_parts)
    max_lora_rank = max(
        [
            checkpoint.lora_rank
            for checkpoint in args.checkpoint_deploy.checkpoint_details.checkpoints
        ]
    )

    start_command_args = {
        "base_model_id": args.checkpoint_deploy.checkpoint_details.base_model_id,
        "lora_modules": checkpoint_str,
        "envvars": start_command_envvars,
        "max_lora_rank": max_lora_rank,
    }
    deploy_config.docker_server.start_command = VLLM_LORA_START_COMMAND.render(
        **start_command_args
    )
    return deploy_config


def get_checkpoint_ids_to_deploy(checkpoint_id_options: List[str]) -> str:
    if len(checkpoint_id_options) == 0:
        raise click.UsageError("No checkpoints found for the training job.")
    if len(checkpoint_id_options) > 1:
        checkpoint_ids = inquirer.checkbox(
            message="Select the checkpoint to deploy. Use spacebar to select/deselect.",
            choices=checkpoint_id_options,
        ).execute()
        if not checkpoint_ids:
            raise click.UsageError("At least one checkpoint must be selected.")
    else:
        checkpoint_ids = [checkpoint_id_options[0]]
    return checkpoint_ids


def hydrate_checkpoints(
    checkpoint_id: str, response_checkpoints: OrderedDict[str, dict]
) -> Checkpoint:
    if checkpoint_id == "latest":
        checkpoint_id = list(response_checkpoints.keys())[-1]
    checkpoint = response_checkpoints[checkpoint_id]
    lora_adapter_config = checkpoint.get("lora_adapter_config") or {}
    max_lora_rank = lora_adapter_config.get("r") or DEFAULT_LORA_RANK
    return Checkpoint(id=checkpoint_id, name=checkpoint_id, lora_rank=max_lora_rank)


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


def get_accelerator_if_specified(
    user_input: Optional[truss_config.AcceleratorSpec],
) -> Optional[truss_config.AcceleratorSpec]:
    if user_input:
        return user_input
    # prompt user for accelerator
    gpu_type = inquirer.select(
        message="Select the GPU type to use for deployment. Select None for CPU.",
        choices=[x.value for x in truss_config.Accelerator] + [None],
    ).execute()
    if gpu_type is None:
        return None
    count = inquirer.text(
        message="Enter the number of accelerators to use for deployment.",
        default="1",
        validate=lambda x: x.isdigit() and int(x) > 0 and int(x) <= 8,
    ).execute()
    return truss_config.AcceleratorSpec(accelerator=gpu_type, count=int(count))


def get_base_model_id(user_input: Optional[str], checkpoint: dict) -> str:
    if user_input:
        return user_input
        # prompt user for base model id
    base_model_id = None
    if base_model_id := checkpoint.get("base_model"):
        rich.print(
            Text(
                f"Inferring base model from checkpoint: {base_model_id}", style="yellow"
            )
        )
    else:
        base_model_id = inquirer.text(message="Enter the base model id.").execute()
    if not base_model_id:
        raise click.UsageError(
            "Base model id is required. Use --base-model-id to specify."
        )
    return base_model_id
