import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, cast

import rich_click as click

import truss.cli.train.core as train_cli
from truss.base.constants import TRAINING_TEMPLATE_DIR
from truss.cli import remote_cli
from truss.cli.cli import truss_cli
from truss.cli.logs import utils as cli_log_utils
from truss.cli.logs.training_log_watcher import TrainingLogWatcher
from truss.cli.train import common as train_common
from truss.cli.train import core
from truss.cli.train.core import (
    SORT_BY_FILEPATH,
    SORT_BY_MODIFIED,
    SORT_BY_PERMISSIONS,
    SORT_BY_SIZE,
    SORT_BY_TYPE,
    SORT_ORDER_ASC,
    SORT_ORDER_DESC,
)
from truss.cli.train.types import DeploySuccessResult
from truss.cli.utils import common
from truss.cli.utils.output import console, error_console
from truss.remote.baseten.core import get_training_job_logs_with_pagination
from truss.remote.baseten.remote import BasetenRemote
from truss.remote.remote_factory import RemoteFactory
from truss.util.path import copy_tree_path
from truss_train import TrainingJob


@click.group()
def train():
    """Subcommands for truss train"""


truss_cli.add_command(train)


def _print_training_job_success_message(
    job_id: str,
    project_id: str,
    project_name: str,
    job_object: Optional[TrainingJob],
    remote_provider: BasetenRemote,
) -> None:
    """Print success message and helpful commands for a training job."""
    console.print("✨ Training job successfully created!", style="green")
    should_print_cache_summary = job_object and (
        job_object.runtime.enable_cache
        or job_object.runtime.cache_config
        and job_object.runtime.cache_config.enabled
    )
    cache_summary_snippet = ""
    if should_print_cache_summary:
        cache_summary_snippet = (
            f"📁 View cache summary via "
            f"[cyan]'truss train cache summarize \"{project_name}\"'[/cyan]\n"
        )
    console.print(
        f"🪵 View logs for your job via "
        f"[cyan]'truss train logs --job-id {job_id} --tail'[/cyan]\n"
        f"🔍 View metrics for your job via "
        f"[cyan]'truss train metrics --job-id {job_id}'[/cyan]\n"
        f"{cache_summary_snippet}"
        f"🌐 View job in the UI: {common.format_link(core.status_page_url(remote_provider.remote_url, project_id, job_id))}"
    )


def _handle_post_create_logic(
    job_resp: dict, remote_provider: BasetenRemote, tail: bool
) -> None:
    project_id, job_id = job_resp["training_project"]["id"], job_resp["id"]
    project_name = job_resp["training_project"]["name"]

    if job_resp.get("current_status", None) == "TRAINING_JOB_QUEUED":
        console.print(
            f"🟢 Training job is queued. You can check the status of your job by running 'truss train view --job-id={job_id}'.",
            style="green",
        )
    else:
        # recreate currently doesn't pass back a job object.
        _print_training_job_success_message(
            job_id,
            project_id,
            project_name,
            job_resp.get("job_object"),
            remote_provider,
        )

    if tail:
        watcher = TrainingLogWatcher(remote_provider.api, project_id, job_id)
        for log in watcher.watch():
            cli_log_utils.output_log(log)


def _prepare_click_context(f: click.Command, params: dict) -> click.Context:
    """create new click context for invoking a command via f.invoke(ctx)"""
    current_ctx = click.get_current_context()
    current_obj = current_ctx.find_root().obj

    ctx = click.Context(f, obj=current_obj)
    ctx.params = params
    return ctx


@train.command(name="push")
@click.argument("config", type=Path, required=True)
@click.option("--remote", type=str, required=False, help="Remote to use")
@click.option("--tail", is_flag=True, help="Tail for status + logs after push.")
@click.option("--job-name", type=str, required=False, help="Name of the training job.")
@common.common_options()
def push_training_job(
    config: Path, remote: Optional[str], tail: bool, job_name: Optional[str]
):
    """Run a training job"""
    from truss_train import deployment

    if not remote:
        remote = remote_cli.inquire_remote_name()

    with console.status("Creating training job...", spinner="dots"):
        remote_provider: BasetenRemote = cast(
            BasetenRemote, RemoteFactory.create(remote=remote)
        )
        job_resp = deployment.create_training_job_from_file(
            remote_provider, config, job_name
        )

    # Note: This post create logic needs to happen outside the context
    # of the above context manager, as only one console session can be active
    # at a time.
    _handle_post_create_logic(job_resp, remote_provider, tail)


@train.command(name="recreate")
@click.option(
    "--job-id", type=str, required=False, help="Job ID of Training Job to recreate"
)
@click.option("--remote", type=str, required=False, help="Remote to use")
@click.option("--tail", is_flag=True, help="Tail for status + logs after recreation.")
@common.common_options()
def recreate_training_job(job_id: Optional[str], remote: Optional[str], tail: bool):
    """Recreate an existing training job from an existing job ID"""
    if not remote:
        remote = remote_cli.inquire_remote_name()

    remote_provider: BasetenRemote = cast(
        BasetenRemote, RemoteFactory.create(remote=remote)
    )

    console.print("Recreating training job...", style="bold")
    job_resp = train_cli.recreate_training_job(
        remote_provider=remote_provider, job_id=job_id
    )

    _handle_post_create_logic(job_resp, remote_provider, tail)


@train.command(name="logs")
@click.option("--remote", type=str, required=False, help="Remote to use")
@click.option("--project-id", type=str, required=False, help="Project ID.")
@click.option("--project", type=str, required=False, help="Project name or project id.")
@click.option("--job-id", type=str, required=False, help="Job ID.")
@click.option("--tail", is_flag=True, help="Tail for ongoing logs.")
@common.common_options()
def get_job_logs(
    remote: Optional[str],
    project_id: Optional[str],
    project: Optional[str],
    job_id: Optional[str],
    tail: bool,
):
    """Fetch logs for a training job"""

    if not remote:
        remote = remote_cli.inquire_remote_name()

    remote_provider: BasetenRemote = cast(
        BasetenRemote, RemoteFactory.create(remote=remote)
    )
    project_id = _maybe_resolve_project_id_from_id_or_name(
        remote_provider, project_id=project_id, project=project
    )

    project_id, job_id = train_common.get_most_recent_job(
        remote_provider, project_id, job_id
    )

    if not tail:
        logs = get_training_job_logs_with_pagination(
            remote_provider.api, project_id, job_id
        )
        for log in cli_log_utils.parse_logs(logs):
            cli_log_utils.output_log(log)
    else:
        log_watcher = TrainingLogWatcher(remote_provider.api, project_id, job_id)
        for log in log_watcher.watch():
            cli_log_utils.output_log(log)


@train.command(name="stop")
@click.option("--project-id", type=str, required=False, help="Project ID.")
@click.option("--project", type=str, required=False, help="Project name or project id.")
@click.option("--job-id", type=str, required=False, help="Job ID.")
@click.option("--all", is_flag=True, help="Stop all running jobs.")
@click.option("--remote", type=str, required=False, help="Remote to use")
@common.common_options()
def stop_job(
    project_id: Optional[str],
    project: Optional[str],
    job_id: Optional[str],
    all: bool,
    remote: Optional[str],
):
    """Stop a training job"""

    if not remote:
        remote = remote_cli.inquire_remote_name()

    remote_provider: BasetenRemote = cast(
        BasetenRemote, RemoteFactory.create(remote=remote)
    )
    project_id = _maybe_resolve_project_id_from_id_or_name(
        remote_provider, project_id=project_id, project=project
    )
    if all:
        train_cli.stop_all_jobs(remote_provider, project_id)
    else:
        project_id, job_id = train_cli.get_args_for_stop(
            remote_provider, project_id, job_id
        )
        remote_provider.api.stop_training_job(project_id, job_id)
        console.print("Training job stopped successfully.", style="green")


@train.command(name="view")
@click.option(
    "--project-id", type=str, required=False, help="View training jobs for a project."
)
@click.option("--project", type=str, required=False, help="Project name or project id.")
@click.option(
    "--job-id", type=str, required=False, help="View a specific training job."
)
@click.option("--remote", type=str, required=False, help="Remote to use")
@common.common_options()
def view_training(
    project_id: Optional[str],
    project: Optional[str],
    job_id: Optional[str],
    remote: Optional[str],
):
    """List all training jobs for a project"""

    if not remote:
        remote = remote_cli.inquire_remote_name()

    remote_provider: BasetenRemote = cast(
        BasetenRemote, RemoteFactory.create(remote=remote)
    )
    project_id = _maybe_resolve_project_id_from_id_or_name(
        remote_provider, project_id=project_id, project=project
    )

    train_cli.view_training_details(remote_provider, project_id, job_id)


@train.command(name="metrics")
@click.option("--project-id", type=str, required=False, help="Project ID.")
@click.option("--project", type=str, required=False, help="Project name or project id.")
@click.option("--job-id", type=str, required=False, help="Job ID.")
@click.option("--remote", type=str, required=False, help="Remote to use")
@common.common_options()
def get_job_metrics(
    project_id: Optional[str],
    project: Optional[str],
    job_id: Optional[str],
    remote: Optional[str],
):
    """Get metrics for a training job"""

    if not remote:
        remote = remote_cli.inquire_remote_name()

    remote_provider: BasetenRemote = cast(
        BasetenRemote, RemoteFactory.create(remote=remote)
    )
    project_id = _maybe_resolve_project_id_from_id_or_name(
        remote_provider, project_id=project_id, project=project
    )
    train_cli.view_training_job_metrics(remote_provider, project_id, job_id)


@train.command(name="deploy_checkpoints")
@click.option("--project-id", type=str, required=False, help="Project ID.")
@click.option("--project", type=str, required=False, help="Project name or project id.")
@click.option("--job-id", type=str, required=False, help="Job ID.")
@click.option(
    "--config",
    type=str,
    required=False,
    help="path to a python file that defines a DeployCheckpointsConfig",
)
@click.option(
    "--dry-run", is_flag=True, help="Generate a truss config without deploying"
)
@click.option(
    "--truss-config-output-dir",
    type=str,
    required=False,
    help="Path to output the truss config to. If not provided, will output to truss_configs/<model_version_name>_<model_version_id> or truss_configs/dry_run_<timestamp> if dry run.",
)
@click.option("--remote", type=str, required=False, help="Remote to use")
@common.common_options()
def deploy_checkpoints(
    project_id: Optional[str],
    project: Optional[str],
    job_id: Optional[str],
    config: Optional[str],
    remote: Optional[str],
    dry_run: bool,
    truss_config_output_dir: Optional[str],
):
    """
    Deploy a LoRA checkpoint via vLLM.
    """

    if not remote:
        remote = remote_cli.inquire_remote_name()

    remote_provider: BasetenRemote = cast(
        BasetenRemote, RemoteFactory.create(remote=remote)
    )
    project_id = _maybe_resolve_project_id_from_id_or_name(
        remote_provider, project_id=project_id, project=project
    )
    result = train_cli.create_model_version_from_inference_template(
        remote_provider,
        train_cli.DeployCheckpointArgs(
            project_id=project_id,
            job_id=job_id,
            deploy_config_path=config,
            dry_run=dry_run,
        ),
    )

    if dry_run:
        console.print("did not deploy because --dry-run flag provided", style="yellow")

    _write_truss_config(result, truss_config_output_dir, dry_run)

    if not dry_run:
        train_cli.print_deploy_checkpoints_success_message(result.deploy_config)


def _write_truss_config(
    result: DeploySuccessResult, truss_config_output_dir: Optional[str], dry_run: bool
) -> None:
    if not result.truss_config:
        return
    # format: 20251006_123456
    datestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = (
        f"{result.model_version.name}_{result.model_version.id}"
        if result.model_version
        else f"dry_run_{datestamp}"
    )
    output_dir_str = truss_config_output_dir or f"truss_configs/{folder_name}"
    output_dir = Path(output_dir_str)
    output_path = output_dir / "config.yaml"
    os.makedirs(output_dir, exist_ok=True)
    console.print(f"Writing truss config to {output_path}", style="yellow")
    console.print(f"👀 Run `cat {output_path}` to view the truss config", style="green")
    if dry_run:
        console.print(
            f"🚀 Run `cd {output_dir} && truss push --publish` to deploy the truss",
            style="green",
        )
    result.truss_config.write_to_yaml_file(output_path)


@train.command(name="download")
@click.option("--job-id", type=str, required=True, help="Job ID.")
@click.option("--remote", type=str, required=False, help="Remote to use")
@click.option(
    "--target-directory",
    type=click.Path(file_okay=False, dir_okay=True, writable=True, resolve_path=True),
    required=False,
    help="Directory where the file should be downloaded. Defaults to current directory.",
)
@click.option(
    "--no-unzip",
    is_flag=True,
    help="Instructs truss to not unzip the folder upon download.",
)
@common.common_options()
def download_training_job(
    job_id: str, remote: Optional[str], target_directory: Optional[str], no_unzip: bool
) -> None:
    if not job_id:
        error_console.print("Job ID is required")
        sys.exit(1)

    if not remote:
        remote = remote_cli.inquire_remote_name()

    remote_provider: BasetenRemote = cast(
        BasetenRemote, RemoteFactory.create(remote=remote)
    )

    try:
        with console.status(
            "[bold green]Downloading training job data...", spinner="dots"
        ):
            target_path = train_cli.download_training_job_data(
                remote_provider=remote_provider,
                job_id=job_id,
                target_directory=target_directory,
                unzip=not no_unzip,
            )

        console.print(
            f"✨ Training job data downloaded to {target_path}", style="bold green"
        )
    except Exception as e:
        error_console.print(f"Failed to download training job data: {str(e)}")
        sys.exit(1)


@train.command(name="get_checkpoint_urls")
@click.option("--job-id", type=str, required=False, help="Job ID.")
@click.option("--remote", type=str, required=False, help="Remote to use")
@common.common_options()
def download_checkpoint_artifacts(job_id: Optional[str], remote: Optional[str]) -> None:
    if not remote:
        remote = remote_cli.inquire_remote_name()

    remote_provider: BasetenRemote = cast(
        BasetenRemote, RemoteFactory.create(remote=remote)
    )

    try:
        with console.status(
            "[bold green]Retrieving checkpoint artifacts...", spinner="dots"
        ):
            target_path = train_cli.download_checkpoint_artifacts(
                remote_provider=remote_provider, job_id=job_id
            )
        console.print(
            f"✨ Training job checkpoint artifacts downloaded to {target_path}",
            style="bold green",
        )
    except Exception as e:
        error_console.print(f"Failed to download checkpoint artifacts data: {str(e)}")
        sys.exit(1)


@train.command(name="init")
@click.option("--list-examples", is_flag=True, help="List all available examples.")
@click.option("--target-directory", type=str, required=False)
@click.option("--examples", type=str, required=False)
@common.common_options()
def init_training_job(
    list_examples: Optional[bool],
    target_directory: Optional[str],
    examples: Optional[str],
) -> None:
    try:
        if list_examples:
            all_examples = train_cli._get_all_train_init_example_options()
            console.print("Available training examples:", style="bold")
            for example in all_examples:
                console.print(f"- {example}")
            console.print(
                "To launch, run `truss train init --examples <example1,example2>`",
                style="bold",
            )
            return

        selected_options = examples.split(",") if examples else []

        # No examples selected, initialize empty training project structure
        if not selected_options:
            if target_directory is None:
                target_directory = "truss-train-init"
            console.print(f"Initializing empty training project at {target_directory}")
            os.makedirs(target_directory)
            copy_tree_path(Path(TRAINING_TEMPLATE_DIR), Path(target_directory))
            console.print(
                f"✨ Empty training project initialized at {target_directory}",
                style="bold green",
            )
            return

        if target_directory is None:
            target_directory = os.getcwd()
        for example_to_download in selected_options:
            download_info = train_cli._get_train_init_example_info(
                example_name=example_to_download
            )
            local_dir = os.path.join(target_directory, example_to_download)

            if not download_info:
                all_examples = train_cli._get_all_train_init_example_options()
                error_console.print(
                    f"Example {example_to_download} not found in the ml-cookbook repository. Examples have to be one or more comma separated values from: {', '.join(all_examples)}"
                )
                continue
            success = train_cli.download_git_directory(
                git_api_url=download_info[0]["url"], local_dir=local_dir
            )
            if success:
                console.print(
                    f"✨ Training directory for {example_to_download} initialized at {local_dir}",
                    style="bold green",
                )
            else:
                error_console.print(
                    f"Failed to initialize training artifacts to {local_dir}"
                )

    except Exception as e:
        error_console.print(f"Failed to initialize training artifacts: {str(e)}")
        sys.exit(1)


@train.group(name="cache")
def cache():
    """Cache-related subcommands for truss train"""


@cache.command(name="summarize")
@click.argument("project", type=str, required=True)
@click.option("--remote", type=str, required=False, help="Remote to use")
@click.option(
    "--sort",
    type=click.Choice(
        [
            SORT_BY_FILEPATH,
            SORT_BY_SIZE,
            SORT_BY_MODIFIED,
            SORT_BY_TYPE,
            SORT_BY_PERMISSIONS,
        ]
    ),
    default=SORT_BY_FILEPATH,
    help="Sort files by filepath, size, modified date, file type, or permissions.",
)
@click.option(
    "--order",
    type=click.Choice([SORT_ORDER_ASC, SORT_ORDER_DESC]),
    default=SORT_ORDER_ASC,
    help="Sort order: ascending or descending.",
)
@common.common_options()
def view_cache_summary(project: str, remote: Optional[str], sort: str, order: str):
    """View cache summary for a training project"""
    if not remote:
        remote = remote_cli.inquire_remote_name()

    remote_provider: BasetenRemote = cast(
        BasetenRemote, RemoteFactory.create(remote=remote)
    )

    train_cli.view_cache_summary_by_project(remote_provider, project, sort, order)


def _maybe_resolve_project_id_from_id_or_name(
    remote_provider: BasetenRemote, project_id: Optional[str], project: Optional[str]
) -> Optional[str]:
    """resolve the project_id or project. `project` can be name or id"""
    if project and project_id:
        console.print("Both `project-id` and `project` provided. Using `project`.")
    project_str = project or project_id
    if not project_str:
        return None
    return train_cli.fetch_project_by_name_or_id(remote_provider, project_str)["id"]
