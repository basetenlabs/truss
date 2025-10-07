import base64
import json
import os
import tarfile
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import click
import requests
import rich
from InquirerPy import inquirer
from rich.text import Text

from truss.cli.train import common, deploy_checkpoints
from truss.cli.train.metrics_watcher import MetricsWatcher
from truss.cli.train.types import (
    DeployCheckpointArgs,
    DeployCheckpointsConfigComplete,
    DeploySuccessResult,
)
from truss.cli.utils import common as cli_common
from truss.cli.utils.output import console
from truss.remote.baseten.custom_types import (
    FileSummary,
    FileSummaryWithTotalSize,
    GetCacheSummaryResponseV1,
)
from truss.remote.baseten.remote import BasetenRemote
from truss_train import loader
from truss_train.definitions import DeployCheckpointsConfig

SORT_BY_FILEPATH = "filepath"
SORT_BY_SIZE = "size"
SORT_BY_MODIFIED = "modified"
SORT_BY_TYPE = "type"
SORT_BY_PERMISSIONS = "permissions"
SORT_ORDER_ASC = "asc"
SORT_ORDER_DESC = "desc"

ACTIVE_JOB_STATUSES = [
    "TRAINING_JOB_RUNNING",
    "TRAINING_JOB_CREATED",
    "TRAINING_JOB_DEPLOYING",
]


def _get_job_by_job_id(remote_provider: BasetenRemote, job_id: str) -> dict:
    jobs = remote_provider.api.search_training_jobs(job_id=job_id)
    if not jobs:
        raise RuntimeError(f"No training job found with ID: {job_id}")
    return jobs[0]


def _get_latest_job(remote_provider: BasetenRemote) -> dict:
    jobs = remote_provider.api.search_training_jobs(
        order_by=[{"field": "created_at", "order": "desc"}]
    )
    if not jobs:
        raise click.ClickException("No training jobs found. Please start a job first.")
    return jobs[0]


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


def recreate_training_job(
    remote_provider: BasetenRemote, job_id: Optional[str] = None
) -> Dict[str, Any]:
    job: dict
    if job_id:
        job = _get_job_by_job_id(remote_provider, job_id)
    else:
        job = _get_latest_job(remote_provider)
        job_id = job["id"]
        confirm = inquirer.confirm(
            message=f"Recreate training job from most recent job {job_id}?",
            default=False,
        ).execute()

        if not confirm:
            raise click.UsageError("Training job not recreated.")

    project_id = job["training_project"]["id"]
    job_id = job["id"]
    job_resp = remote_provider.api.recreate_training_job(project_id, job_id)  # type: ignore
    return job_resp


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
                status_page_url(remote_url, project["id"], latest_job_id), "link"
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


def create_model_version_from_inference_template(
    remote_provider: BasetenRemote, args: DeployCheckpointArgs
) -> DeploySuccessResult:
    if not args.deploy_config_path:
        return deploy_checkpoints.create_model_version_from_inference_template(
            remote_provider,
            DeployCheckpointsConfig(),
            args.project_id,
            args.job_id,
            args.dry_run,
        )
    #### User provided a checkpoint deploy config file
    with loader.import_deploy_checkpoints_config(
        Path(args.deploy_config_path)
    ) as checkpoint_deploy:
        return deploy_checkpoints.create_model_version_from_inference_template(
            remote_provider,
            checkpoint_deploy,
            args.project_id,
            args.job_id,
            args.dry_run,
        )


def _get_checkpoint_names(
    checkpoint_deploy_config: DeployCheckpointsConfigComplete,
) -> list[str]:
    return [
        checkpoint.checkpoint_name
        for checkpoint in checkpoint_deploy_config.checkpoint_details.checkpoints
    ]


def print_deploy_checkpoints_success_message(
    checkpoint_deploy_config: DeployCheckpointsConfigComplete,
):
    checkpoint_names = _get_checkpoint_names(checkpoint_deploy_config)
    console.print(
        Text("\nTo run the model"),
        Text("ensure your `model` parameter is set to one of"),
        Text(
            f"{[checkpoint_name for checkpoint_name in checkpoint_names]}",
            style="magenta",
        ),
        Text("in your request. An example request body might look like this:"),
        Text(
            f'\n{{"model": "{checkpoint_names[0]}", "messages": [...]}}', style="green"
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
    table.add_row("Job Name", job["name"])
    table.add_row("Job ID", job["id"])
    table.add_row("Project ID", job["training_project"]["id"])
    table.add_row("Project Name", job["training_project"]["name"])
    table.add_row("Status", job["current_status"])
    table.add_row("Instance Type", job["instance_type"]["name"])
    table.add_row("Created", cli_common.format_localized_time(job["created_at"]))
    table.add_row("Last Modified", cli_common.format_localized_time(job["updated_at"]))
    table.add_row(
        "Job Page",
        cli_common.format_link(
            status_page_url(remote_url, job["training_project"]["id"], job["id"]),
            "link",
        ),
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


def _generate_job_artifact_name(project_name: str, job_id: str) -> str:
    return f"{project_name}_{job_id}"


def download_training_job_data(
    remote_provider: BasetenRemote,
    job_id: str,
    target_directory: Optional[str],
    unzip: bool,
) -> Path:
    output_dir = Path(target_directory).resolve() if target_directory else Path.cwd()
    output_dir.mkdir(parents=True, exist_ok=True)

    job = _get_job_by_job_id(remote_provider, job_id)

    project = job["training_project"]
    project_id = project["id"]
    project_name = project["name"]

    artifact_base_name = _generate_job_artifact_name(project_name, job_id)
    file_name = f"{artifact_base_name}.tgz"
    target_path = output_dir / file_name

    presigned_url = remote_provider.api.get_training_job_presigned_url(
        project_id=project_id, job_id=job_id
    )
    content = remote_provider.api.get_from_presigned_url(presigned_url)

    if unzip:
        with tempfile.NamedTemporaryFile() as temp_file:
            temp_path = Path(temp_file.name)
            temp_path.write_bytes(content)

            unzip_dir = output_dir / artifact_base_name
            unzip_dir = Path(str(unzip_dir).replace(" ", "-"))
            if unzip_dir.exists():
                raise click.ClickException(
                    f"Directory '{unzip_dir}' already exists. "
                    "Please remove it or specify a different target directory."
                )

            unzip_dir.mkdir(parents=True, exist_ok=True)
            with tarfile.open(temp_path, "r:*") as tar:
                tar.extractall(path=unzip_dir)

            return unzip_dir
    else:
        target_path = Path(str(target_path).replace(" ", "-"))
        target_path.write_bytes(content)
        return target_path


def download_checkpoint_artifacts(
    remote_provider: BasetenRemote, job_id: Optional[str]
) -> Path:
    output_dir = Path.cwd()
    job: dict

    if job_id:
        job = _get_job_by_job_id(remote_provider, job_id)
    else:
        job = _get_latest_job(remote_provider)

    job_id = job["id"]
    project = job["training_project"]
    project_id = project["id"]
    project_name = project["name"]

    checkpoint_artifacts = (
        remote_provider.api.get_training_job_checkpoint_presigned_url(
            project_id=project_id,
            job_id=job_id,  # type: ignore
            page_size=1000,
        )
    )

    if not checkpoint_artifacts:
        raise click.ClickException("No checkpoints found for this training job.")

    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "job": job,
        "checkpoint_artifacts": checkpoint_artifacts,
    }

    urls_file = (
        output_dir / f"{project_name.replace(' ', '-')}_{job_id}_checkpoints.json"
    )
    with open(urls_file, "w") as f:
        json.dump(output, f, indent=2)

    return urls_file


def status_page_url(remote_url: str, project_id: str, training_job_id: str) -> str:
    return f"{remote_url}/training/{project_id}/logs/{training_job_id}"


def _get_all_train_init_example_options(
    repo_id: str = "ml-cookbook",
    examples_subdir: str = "examples",
    token: Optional[str] = None,
) -> list[str]:
    """
    Retrieve a list of all example options from the ml-cookbook repository to
    copy locally for training initialization. This method generates a list
    of examples and URL paths to show the user for selection.
    """
    headers = {}
    if token:
        headers["Authorization"] = f"token {token}"

    url = (
        f"https://api.github.com/repos/basetenlabs/{repo_id}/contents/{examples_subdir}"
    )
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        items = response.json()
        if not isinstance(items, list):
            items = [items]
        items = [item["name"] for item in items if item["type"] == "dir"]
        return items

    except requests.exceptions.RequestException as e:
        click.echo(
            f"Error exploring directory: {e}. Please file an issue at https://github.com/basetenlabs/truss/issues"
        )
        return []


def _get_train_init_example_info(
    repo_id: str = "ml-cookbook",
    examples_subdir: str = "examples",
    training_subdir: str = "training",
    example_name: Optional[str] = None,
    token: Optional[str] = None,
) -> list[Dict[str, str]]:
    """
    Retrieve directory download links for the example from the ml-cookbook repository to
    copy locally for training initialization.
    """
    headers = {}
    if token:
        headers["Authorization"] = f"token {token}"

    url = f"https://api.github.com/repos/basetenlabs/{repo_id}/contents/{examples_subdir}/{example_name}"

    console.print(f"Attempting to retrieve example info from: {url}")
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        items = response.json()
        if not isinstance(items, list):
            items = [items]
        # return only training subdirectory info for example
        items = [item for item in items if item["name"] == training_subdir]
        return items

    except requests.exceptions.HTTPError as e:
        if response.status_code == 404:
            # example_name does not exist, return empty list
            return []
        else:
            # Other HTTP errors
            click.echo(
                f"Error exploring directory: {e}. Please file an issue at https://github.com/basetenlabs/truss/issues"
            )
            return []
    except requests.exceptions.RequestException as e:
        # Network or other request errors
        click.echo(
            f"Error exploring directory: {e}. Please file an issue at https://github.com/basetenlabs/truss/issues"
        )
        return []


def download_git_directory(
    git_api_url: str, local_dir: str, token: Optional[str] = None
):
    """
    Recursively download directory contents from git api url.
    Special handling for 'training' directory: downloads its contents directly
    to local_dir without creating a 'training' subdirectory.
    Args:
        git_api_url (str): Example format "https://api.github.com/repos/basetenlabs/ml-cookbook/contents/examples/llama-finetune-8b-lora?ref=main"
        local_dir(str): Local directory to download this directory to
    """
    headers = {}
    if token:
        headers["Authorization"] = f"token {token}"
    try:
        response = requests.get(git_api_url, headers=headers)
        response.raise_for_status()
        items = response.json()

        # Handle single file case
        if not isinstance(items, list):
            items = [items]

        # Create local directory
        print(f"Creating directory {local_dir}")
        os.makedirs(local_dir, exist_ok=True)

        # Check if there's a 'training' directory in the items
        training_dir = None
        other_items = []

        for item in items:
            if item["name"] == "training" and item["type"] == "dir":
                training_dir = item
            else:
                other_items.append(item)

        # If training directory exists, download its contents directly to local_dir
        if training_dir:
            print(
                f"📁 Found training directory, downloading its contents to {local_dir}"
            )
            return download_git_directory(training_dir["url"], local_dir)

        # If no training directory, download all files normally
        for item in other_items:
            item_name = item["name"]
            local_item_path = os.path.join(local_dir, item_name)

            if item["type"] == "file":
                print(f"📄 Downloading {item_name}")
                if item.get("download_url"):
                    # Download file directly
                    file_response = requests.get(item["download_url"])
                    file_response.raise_for_status()
                    with open(local_item_path, "wb") as f:
                        f.write(file_response.content)
                elif item.get("content"):
                    # Decode base64 content (for small files)
                    try:
                        content = base64.b64decode(item["content"])
                        with open(local_item_path, "wb") as f:
                            f.write(content)
                    except Exception as e:
                        print(f"⚠️ Could not decode {item_name}: {e}")
            elif item["type"] == "dir":
                print(f"📁 Entering directory {item_name}")
                # Use the API URL from the response for subdirectories
                download_git_directory(item["url"], local_item_path)
        return True
    except Exception as e:
        print(f"Error processing response: {e}")
        return False


def fetch_project_by_name_or_id(
    remote_provider: BasetenRemote, project_identifier: str
) -> dict:
    """Fetch a training project by name or ID.

    Args:
        remote_provider: The remote provider instance
        project_identifier: Either a project ID or project name

    Returns:
        The project object as a dictionary

    Raises:
        click.ClickException: If the project is not found
    """
    try:
        projects = remote_provider.api.list_training_projects()
        projects_by_name = {project.get("name"): project for project in projects}
        projects_by_id = {project.get("id"): project for project in projects}
        if project_identifier in projects_by_id:
            return projects_by_id[project_identifier]
        if project_identifier in projects_by_name:
            return projects_by_name[project_identifier]
        valid_project_ids_and_names = ", ".join(
            [f"{project.get('id')} ({project.get('name')})" for project in projects]
        )
        raise click.ClickException(
            f"Project '{project_identifier}' not found. Valid project IDs and names: {valid_project_ids_and_names}"
        )
    except click.ClickException:
        raise
    except Exception as e:
        raise click.ClickException(f"Error fetching project: {str(e)}")


def create_file_summary_with_directory_sizes(
    files: list[FileSummary],
) -> list[FileSummaryWithTotalSize]:
    directory_sizes = calculate_directory_sizes(files)
    return [
        FileSummaryWithTotalSize(
            file_summary=file_info,
            total_size=directory_sizes.get(file_info.path, file_info.size_bytes),
        )
        for file_info in files
    ]


def calculate_directory_sizes(
    files: list[FileSummary], max_depth: int = 100
) -> dict[str, int]:
    directory_sizes = {}

    for file_info in files:
        if file_info.file_type == "directory":
            directory_sizes[file_info.path] = 0

    for file_info in files:
        current_path = file_info.path
        for i in range(max_depth):
            if current_path is None:
                break
            if current_path in directory_sizes:
                directory_sizes[current_path] += file_info.size_bytes
            # Move to parent directory
            parent = os.path.dirname(current_path)
            if parent == current_path:  # Reached root
                break
            current_path = parent

    return directory_sizes


def view_cache_summary(
    remote_provider: BasetenRemote,
    project_id: str,
    sort_by: str = SORT_BY_FILEPATH,
    order: str = SORT_ORDER_ASC,
):
    """View cache summary for a training project."""
    try:
        raw_cache_data = remote_provider.api.get_cache_summary(project_id)

        if not raw_cache_data:
            console.print("No cache summary found for this project.", style="yellow")
            return

        cache_data = GetCacheSummaryResponseV1.model_validate(raw_cache_data)

        table = rich.table.Table(title=f"Cache summary for project: {project_id}")
        table.add_column("File Path", style="cyan")
        table.add_column("Size", style="green")
        table.add_column("Modified", style="yellow")
        table.add_column("Type")
        table.add_column("Permissions", style="magenta")

        files = cache_data.file_summaries
        if not files:
            console.print("No files found in cache.", style="yellow")
            return

        files_with_total_sizes = create_file_summary_with_directory_sizes(files)

        reverse = order == SORT_ORDER_DESC
        sort_key = _get_sort_key(sort_by)
        files_with_total_sizes.sort(key=sort_key, reverse=reverse)

        total_size = sum(
            file_info.file_summary.size_bytes for file_info in files_with_total_sizes
        )
        total_size_str = common.format_bytes_to_human_readable(total_size)

        console.print(
            f"📅 Cache captured at: {cache_data.timestamp}", style="bold blue"
        )
        console.print(f"📁 Project ID: {cache_data.project_id}", style="bold blue")
        console.print()
        console.print(
            f"📊 Total files: {len(files_with_total_sizes)}", style="bold green"
        )
        console.print(f"💾 Total size: {total_size_str}", style="bold green")
        console.print()

        for file_info in files_with_total_sizes:
            total_size = file_info.total_size

            size_str = cli_common.format_bytes_to_human_readable(int(total_size))

            modified_str = cli_common.format_localized_time(
                file_info.file_summary.modified
            )

            table.add_row(
                file_info.file_summary.path,
                size_str,
                modified_str,
                file_info.file_summary.file_type or "Unknown",
                file_info.file_summary.permissions or "Unknown",
            )

        console.print(table)

    except Exception as e:
        console.print(f"Error fetching cache summary: {str(e)}", style="red")
        raise


def _get_sort_key(sort_by: str) -> Callable[[FileSummaryWithTotalSize], Any]:
    if sort_by == SORT_BY_FILEPATH:
        return lambda x: x.file_summary.path
    elif sort_by == SORT_BY_SIZE:
        return lambda x: x.total_size
    elif sort_by == SORT_BY_MODIFIED:
        return lambda x: x.file_summary.modified
    elif sort_by == SORT_BY_TYPE:
        return lambda x: x.file_summary.file_type or ""
    elif sort_by == SORT_BY_PERMISSIONS:
        return lambda x: x.file_summary.permissions or ""
    else:
        raise ValueError(f"Invalid --sort argument: {sort_by}")


def view_cache_summary_by_project(
    remote_provider: BasetenRemote,
    project_identifier: str,
    sort_by: str = SORT_BY_FILEPATH,
    order: str = SORT_ORDER_ASC,
):
    """View cache summary for a training project by ID or name."""
    project = fetch_project_by_name_or_id(remote_provider, project_identifier)
    view_cache_summary(remote_provider, project["id"], sort_by, order)
