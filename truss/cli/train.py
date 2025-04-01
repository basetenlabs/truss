from typing import Optional, Tuple, cast

import rich
import rich_click as click
from InquirerPy import inquirer
from rich.console import Console

from truss.remote.baseten.remote import BasetenRemote


def get_args_for_stop(
    console: Console,
    remote_provider: BasetenRemote,
    project_id: Optional[str],
    job_id: Optional[str],
) -> Tuple[str, str]:
    if not project_id or not job_id:
        # get all running jobs
        job = _get_running_job(console, remote_provider, project_id, job_id)
        project_id_for_job = cast(str, job["training_project_id"])
        job_id_to_stop = cast(str, job["id"])
        # check if the user wants to stop the inferred running job
        if not job_id:
            confirm = inquirer.confirm(
                message=f"Are you sure you want to stop training job {job_id_to_stop}?",
                default=False,
            ).execute()
            if not confirm:
                raise click.UsageError("Training job not stopped.")
        project_id = project_id_for_job
        job_id = job_id_to_stop

    return project_id, job_id


def get_args_for_logs(
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
            raise click.UsageError("Unable to get logs. No jobs found.")
        if len(jobs) > 1:
            sorted_jobs = sorted(jobs, key=lambda x: x["created_at"], reverse=True)
            job = sorted_jobs[0]
            console.print(
                f"Multiple jobs found. Showing logs for the most recently created job: {job['id']}",
                style="yellow",
            )
        else:
            job = jobs[0]
        project_id = cast(str, job["training_project_id"])
        job_id = cast(str, job["id"])
    return project_id, job_id


def _get_running_job(
    console: Console,
    remote_provider: BasetenRemote,
    project_id: Optional[str],
    job_id: Optional[str],
) -> dict:
    jobs = remote_provider.api.search_training_jobs(
        statuses=["TRAINING_JOB_RUNNING"], project_id=project_id, job_id=job_id
    )
    if not jobs:
        raise click.UsageError("No running jobs found.")
    if len(jobs) > 1:
        display_training_jobs(console, jobs, title="Running Training Jobs")
        raise click.UsageError("Multiple running jobs found. Please specify a job id.")
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
    table.add_column("Job ID", style="cyan")
    table.add_column("Status", style="white")
    table.add_column("Instance Type", style="white")
    table.add_column("Created At", style="white")
    table.add_column("Updated At", style="white")
    for job in jobs:
        table.add_row(
            job["training_project_id"],
            job["id"],
            job["current_status"],
            job["instance_type"]["name"],
            job["created_at"],
            job["updated_at"],
        )
    console.print(table)
