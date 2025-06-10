import sys
from pathlib import Path
from typing import Optional, cast

import rich_click as click

import truss.cli.train.core as train_cli
from truss.cli import remote_cli
from truss.cli.cli import push, truss_cli
from truss.cli.logs import utils as cli_log_utils
from truss.cli.logs.training_log_watcher import TrainingLogWatcher
from truss.cli.train import common as train_common
from truss.cli.train import core
from truss.cli.utils import common
from truss.cli.utils.output import console, error_console
from truss.remote.baseten.remote import BasetenRemote
from truss.remote.remote_factory import RemoteFactory


@click.group()
def train():
    """Subcommands for truss train"""


truss_cli.add_command(train)


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
@common.common_options()
def push_training_job(config: Path, remote: Optional[str], tail: bool):
    """Run a training job"""
    from truss_train import deployment, loader

    if not remote:
        remote = remote_cli.inquire_remote_name()

    remote_provider: BasetenRemote = cast(
        BasetenRemote, RemoteFactory.create(remote=remote)
    )
    with loader.import_training_project(config) as training_project:
        with console.status("Creating training job...", spinner="dots"):
            project_resp = remote_provider.api.upsert_training_project(
                training_project=training_project
            )

            prepared_job = deployment.prepare_push(
                remote_provider.api, config, training_project.job
            )
            job_resp = remote_provider.api.create_training_job(
                project_id=project_resp["id"], job=prepared_job
            )

        job_id = job_resp["id"]

        console.print("✨ Training job successfully created!", style="green")
        console.print(
            f"🪵 View logs for your job via "
            f"[cyan]'truss train logs --job-id {job_id} [--tail]'[/cyan]\n"
            f"🔍 View metrics for your job via "
            f"[cyan]'truss train metrics --job-id {job_id}'[/cyan]\n"
            f"🌐 Status page: {common.format_link(core.status_page_url(remote_provider.remote_url, job_id))}"
        )

    if tail:
        project_id, job_id = project_resp["id"], job_resp["id"]
        watcher = TrainingLogWatcher(remote_provider.api, project_id, job_id)
        for log in watcher.watch():
            cli_log_utils.output_log(log)


@train.command(name="logs")
@click.option("--remote", type=str, required=False, help="Remote to use")
@click.option("--project-id", type=str, required=False, help="Project ID.")
@click.option("--job-id", type=str, required=False, help="Job ID.")
@click.option("--tail", is_flag=True, help="Tail for ongoing logs.")
@common.common_options()
def get_job_logs(
    remote: Optional[str], project_id: Optional[str], job_id: Optional[str], tail: bool
):
    """Fetch logs for a training job"""

    if not remote:
        remote = remote_cli.inquire_remote_name()

    remote_provider: BasetenRemote = cast(
        BasetenRemote, RemoteFactory.create(remote=remote)
    )
    project_id, job_id = train_common.get_most_recent_job(
        remote_provider, project_id, job_id
    )

    if not tail:
        logs = remote_provider.api.get_training_job_logs(project_id, job_id)
        for log in cli_log_utils.parse_logs(logs):
            cli_log_utils.output_log(log)
    else:
        log_watcher = TrainingLogWatcher(remote_provider.api, project_id, job_id)
        for log in log_watcher.watch():
            cli_log_utils.output_log(log)


@train.command(name="stop")
@click.option("--project-id", type=str, required=False, help="Project ID.")
@click.option("--job-id", type=str, required=False, help="Job ID.")
@click.option("--all", is_flag=True, help="Stop all running jobs.")
@click.option("--remote", type=str, required=False, help="Remote to use")
@common.common_options()
def stop_job(
    project_id: Optional[str], job_id: Optional[str], all: bool, remote: Optional[str]
):
    """Stop a training job"""

    if not remote:
        remote = remote_cli.inquire_remote_name()

    remote_provider: BasetenRemote = cast(
        BasetenRemote, RemoteFactory.create(remote=remote)
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
@click.option(
    "--job-id", type=str, required=False, help="View a specific training job."
)
@click.option("--remote", type=str, required=False, help="Remote to use")
@common.common_options()
def view_training(
    project_id: Optional[str], job_id: Optional[str], remote: Optional[str]
):
    """List all training jobs for a project"""

    if not remote:
        remote = remote_cli.inquire_remote_name()

    remote_provider: BasetenRemote = cast(
        BasetenRemote, RemoteFactory.create(remote=remote)
    )
    train_cli.view_training_details(remote_provider, project_id, job_id)


@train.command(name="metrics")
@click.option("--project-id", type=str, required=False, help="Project ID.")
@click.option("--job-id", type=str, required=False, help="Job ID.")
@click.option("--remote", type=str, required=False, help="Remote to use")
@common.common_options()
def get_job_metrics(
    project_id: Optional[str], job_id: Optional[str], remote: Optional[str]
):
    """Get metrics for a training job"""

    if not remote:
        remote = remote_cli.inquire_remote_name()

    remote_provider: BasetenRemote = cast(
        BasetenRemote, RemoteFactory.create(remote=remote)
    )
    train_cli.view_training_job_metrics(remote_provider, project_id, job_id)


@train.command(name="deploy_checkpoints")
@click.option("--project-id", type=str, required=False, help="Project ID.")
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
@click.option("--remote", type=str, required=False, help="Remote to use")
@common.common_options()
def deploy_checkpoints(
    project_id: Optional[str],
    job_id: Optional[str],
    config: Optional[str],
    remote: Optional[str],
    dry_run: bool,
):
    """
    Deploy a LoRA checkpoint via vLLM.
    """

    if not remote:
        remote = remote_cli.inquire_remote_name()

    remote_provider: BasetenRemote = cast(
        BasetenRemote, RemoteFactory.create(remote=remote)
    )
    prepare_checkpoint_result = train_cli.prepare_checkpoint_deploy(
        remote_provider,
        train_cli.PrepareCheckpointArgs(
            project_id=project_id, job_id=job_id, deploy_config_path=config
        ),
    )

    params = {
        "target_directory": prepare_checkpoint_result.truss_directory,
        "remote": remote,
        "model_name": prepare_checkpoint_result.checkpoint_deploy_config.model_name,
        "publish": True,
        "deployment_name": prepare_checkpoint_result.checkpoint_deploy_config.deployment_name,
    }
    ctx = _prepare_click_context(push, params)
    if dry_run:
        console.print("--dry-run flag provided, not deploying", style="yellow")
    else:
        push.invoke(ctx)
    train_cli.print_deploy_checkpoints_success_message(prepare_checkpoint_result)


@train.command(name="download")
@click.option("--job-id", type=str, required=True, help="Job ID.")
@click.option("--remote", type=str, required=False, help="Remote to use")
@click.option(
    "--target-directory",
    type=click.Path(file_okay=False, dir_okay=True, writable=True, resolve_path=True),
    required=False,
    help="Directory where the file should be downloaded. Defaults to current directory.",
)
@common.common_options()
def download_training_job(
    job_id: str, remote: Optional[str], target_directory: Optional[str]
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
            )

        console.print(
            f"✨ Training job data downloaded to {target_path}", style="bold green"
        )
    except Exception as e:
        error_console.print(f"Failed to download training job data: {str(e)}")
        sys.exit(1)
