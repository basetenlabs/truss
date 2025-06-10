from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple

import rich
import rich_click as click
from InquirerPy import inquirer
from rich.text import Text

from truss.cli.train import common, deploy_checkpoints
from truss.cli.train.metrics_watcher import MetricsWatcher
from truss.cli.train.types import PrepareCheckpointArgs, PrepareCheckpointResult
from truss.cli.utils import common as cli_common
from truss.cli.utils.output import console
from truss.remote.baseten.remote import BasetenRemote
from truss_train import loader
from truss_train.definitions import DeployCheckpointsConfig

ACTIVE_JOB_STATUSES = [
    "TRAINING_JOB_RUNNING",
    "TRAINING_JOB_CREATED",
    "TRAINING_JOB_DEPLOYING",
]


def get_args_for_stop(
    remote_provider: BasetenRemote, project_id: Optional[str], job_id: Optional[str]
) -> Tuple[str, str]:
    if not project_id or not job_id:
        # get all running jobs
        job = _get_active_job(remote_provider, project_id, job_id)
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


def _get_active_job(
    remote_provider: BasetenRemote, project_id: Optional[str], job_id: Optional[str]
) -> dict:
    jobs = remote_provider.api.search_training_jobs(
        statuses=ACTIVE_JOB_STATUSES, project_id=project_id, job_id=job_id
    )
    if not jobs:
        raise click.UsageError("No running jobs found.")
    if len(jobs) > 1:
        display_training_jobs(
            jobs, remote_provider.remote_url, title="Active Training Jobs"
        )
        raise click.UsageError("Multiple active jobs found. Please specify a job id.")
    return jobs[0]


@dataclass
class DisplayTableColumn:
    name: str
    style: dict
    accessor: Callable[[dict], str]


def display_training_jobs(
    jobs, remote_url: str, checkpoints_by_job_id=None, title="Training Job Details"
):
    checkpoints_by_job_id = checkpoints_by_job_id or {}
    console.print(title, style="bold magenta")
    for job in jobs:
        display_training_job(job, remote_url, checkpoints_by_job_id.get(job["id"], []))


def display_training_projects(projects: list[dict], remote_url: str) -> None:
    table = rich.table.Table(
        show_header=True,
        header_style="bold magenta",
        title="Training Projects",
        box=rich.table.box.ROUNDED,
        border_style="blue",
    )
    table.add_column("Project ID", style="cyan")
    table.add_column("Name")
    table.add_column("Created")
    table.add_column("Last Modified")
    table.add_column("Latest Job ID", style="bold yellow")
    table.add_column("Latest Job Status", style="bold yellow")
    table.add_column("Status Page", style="bold yellow")

    # most recent projects at bottom of terminal
    for project in projects[::-1]:
        latest_job = project.get("latest_job") or {}
        if latest_job_id := latest_job.get("id", ""):
            latest_job_link = cli_common.format_link(
                status_page_url(remote_url, latest_job_id), "link"
            )
        else:
            latest_job_link = ""
        table.add_row(
            project["id"],
            project["name"],
            cli_common.format_localized_time(project["created_at"]),
            cli_common.format_localized_time(project["updated_at"]),
            latest_job_id,
            latest_job.get("current_status", ""),
            latest_job_link,
        )

    console.print(table)


def view_training_details(
    remote_provider: BasetenRemote, project_id: Optional[str], job_id: Optional[str]
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
        if len(jobs_response) == 1:
            training_job = jobs_response[0]
            checkpoints = remote_provider.api.list_training_job_checkpoints(
                training_job["training_project"]["id"], training_job["id"]
            )
            display_training_job(
                training_job, remote_provider.remote_url, checkpoints["checkpoints"]
            )
        else:
            display_training_jobs(jobs_response, remote_provider.remote_url)
    else:
        projects = remote_provider.api.list_training_projects()
        display_training_projects(projects, remote_provider.remote_url)
        active_jobs = remote_provider.api.search_training_jobs(
            statuses=ACTIVE_JOB_STATUSES
        )
        if active_jobs:
            display_training_jobs(
                active_jobs, remote_provider.remote_url, title="Active Training Jobs"
            )
        else:
            console.print("No active training jobs.", style="yellow")


def stop_all_jobs(remote_provider: BasetenRemote, project_id: Optional[str]):
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
    remote_provider: BasetenRemote, project_id: Optional[str], job_id: Optional[str]
):
    """
    view_training_job_metrics shows a list of metrics for a training job.
    """
    project_id, job_id = common.get_most_recent_job(remote_provider, project_id, job_id)
    metrics_display = MetricsWatcher(remote_provider.api, project_id, job_id)
    metrics_display.watch()


def prepare_checkpoint_deploy(
    remote_provider: BasetenRemote, args: PrepareCheckpointArgs
) -> PrepareCheckpointResult:
    if not args.deploy_config_path:
        return deploy_checkpoints.prepare_checkpoint_deploy(
            remote_provider, DeployCheckpointsConfig(), args.project_id, args.job_id
        )
    #### User provided a checkpoint deploy config file
    with loader.import_deploy_checkpoints_config(
        Path(args.deploy_config_path)
    ) as checkpoint_deploy:
        return deploy_checkpoints.prepare_checkpoint_deploy(
            remote_provider, checkpoint_deploy, args.project_id, args.job_id
        )


def print_deploy_checkpoints_success_message(
    prepare_checkpoint_result: PrepareCheckpointResult,
):
    console.print(
        Text("\nTo run the model with the LoRA adapter,"),
        Text("ensure your `model` parameter is set to one of"),
        Text(
            f"{[x.id for x in prepare_checkpoint_result.checkpoint_deploy_config.checkpoint_details.checkpoints]}",
            style="magenta",
        ),
        Text("in your request. An example request body might look like this:"),
        Text(
            "\n{"
            + f'"model": {prepare_checkpoint_result.checkpoint_deploy_config.checkpoint_details.checkpoints[0].id}, "messages": [...]'
            + "}",
            style="green",
        ),
    )


def display_training_job(
    job: dict, remote_url: str, checkpoints: Optional[list[dict]] = None
):
    table = rich.table.Table(
        show_header=False,
        title=f"Training Job: {job['id']}",
        box=rich.table.box.ROUNDED,
        border_style="blue",
        title_style="bold magenta",
    )
    table.add_column("Field", style="bold cyan")
    table.add_column("Value")

    # Basic job details
    table.add_row("Project ID", job["training_project"]["id"])
    table.add_row("Project Name", job["training_project"]["name"])
    table.add_row("Job ID", job["id"])
    table.add_row("Status", job["current_status"])
    table.add_row("Instance Type", job["instance_type"]["name"])
    table.add_row("Created", cli_common.format_localized_time(job["created_at"]))
    table.add_row("Last Modified", cli_common.format_localized_time(job["updated_at"]))
    table.add_row(
        "Status Page",
        cli_common.format_link(status_page_url(remote_url, job["id"]), "link"),
    )

    # Add error message if present
    if job.get("error_message"):
        table.add_row("Error Message", Text(job["error_message"], style="red"))

    # Add checkpoints if present
    if checkpoints:
        checkpoint_text = ", ".join(
            [checkpoint["checkpoint_id"] for checkpoint in checkpoints]
        )
        table.add_row("Checkpoints", checkpoint_text)

    console.print(table)


def download_training_job_data(
    remote_provider: BasetenRemote, job_id: str, target_directory: Optional[str]
) -> Path:
    output_dir = Path(target_directory).resolve() if target_directory else Path.cwd()
    output_dir.mkdir(parents=True, exist_ok=True)

    jobs = remote_provider.api.search_training_jobs(job_id=job_id)

    if not jobs:
        raise RuntimeError(f"No training job found with ID: {job_id}")

    project = jobs[0]["training_project"]
    project_id = project["id"]
    project_name = project["name"]

    file_name = f"{project_name}_{job_id}.tgz"
    target_path = output_dir / file_name

    presigned_url = remote_provider.api.get_training_job_presigned_url(
        project_id=project_id, job_id=job_id
    )

    content = remote_provider.api.get_from_presigned_url(presigned_url)
    target_path.write_bytes(content)

    return target_path


def status_page_url(remote_url: str, training_job_id: str) -> str:
    return f"{remote_url}/training/jobs/{training_job_id}"
