import os
import sys
from typing import Optional, cast

import rich_click as click
from InquirerPy import inquirer

from truss.cli import remote_cli
from truss.cli.cli import truss_cli
from truss.cli.ssh import (
    ensure_ssh_keypair,
    install_proxy_command_script,
    is_setup_complete,
    setup_ssh_config,
)
from truss.cli.train.common import get_most_recent_job
from truss.cli.train.poller import TrainingPollerMixin
from truss.cli.utils import common
from truss.cli.utils.common import check_is_interactive
from truss.cli.utils.output import console
from truss.remote.baseten.remote import BasetenRemote
from truss.remote.remote_factory import RemoteFactory


@click.group(invoke_without_command=True)
@click.option(
    "--training-job-id",
    "training_job_id",
    type=str,
    required=False,
    help="Training job ID to SSH into. Waits for the job to be running, then connects.",
)
@click.option(
    "--node-id",
    "node_id",
    type=int,
    default=0,
    show_default=True,
    help="Node index to connect to for multi-node jobs.",
)
@click.option("--remote", type=str, required=False, help="Remote to use.")
@click.pass_context
def ssh(
    ctx: click.Context,
    training_job_id: Optional[str],
    node_id: int,
    remote: Optional[str],
):
    """SSH access to Baseten workloads.

    Pass --training-job-id <id> to wait for a training job to be running and
    then connect via SSH. Use `truss ssh setup` for one-time SSH setup.
    """
    if ctx.invoked_subcommand is not None:
        return
    if not training_job_id:
        click.echo(ctx.get_help())
        ctx.exit(2)
    _wait_and_exec_ssh_training_job(training_job_id, node_id, remote)


truss_cli.add_command(ssh)


def _wait_and_exec_ssh_training_job(
    job_id: str, node_id: int, remote: Optional[str]
) -> None:
    if not is_setup_complete():
        console.print(
            "SSH is not set up yet. Run [cyan]truss ssh setup[/cyan] first.",
            style="yellow",
        )
        sys.exit(1)

    if node_id < 0:
        raise click.UsageError("--node-id must be >= 0")

    if not remote:
        remote = remote_cli.inquire_remote_name()

    remote_provider = cast(BasetenRemote, RemoteFactory.create(remote=remote))

    project_id, job_id = get_most_recent_job(remote_provider, None, job_id)

    job_resp = remote_provider.api.get_training_job(project_id, job_id)
    training_job = job_resp.get("training_job", {})
    instance_type = training_job.get("instance_type") or {}
    node_count = instance_type.get("node_count") or instance_type.get("nodeCount") or 1
    if node_id >= node_count:
        raise click.UsageError(
            f"--node-id {node_id} is out of range; job has {node_count} node"
            f"{'s' if node_count != 1 else ''} (0..{node_count - 1})."
        )

    poller = TrainingPollerMixin(remote_provider.api, project_id, job_id)
    poller.before_polling()
    status = poller._current_status.status
    if status != "TRAINING_JOB_RUNNING":
        console.print(
            f"Job is not running (status: {status}). Cannot SSH.", style="red"
        )
        sys.exit(1)

    hostname = f"training-job-{job_id}-{node_id}.ssh.baseten.co"
    console.print(f"Connecting to [cyan]{hostname}[/cyan]...")
    os.execvp("ssh", ["ssh", hostname])


@ssh.command(name="setup")
@click.option(
    "--python",
    "python_path",
    type=str,
    required=False,
    help="Path to Python 3.10+ interpreter for the ProxyCommand. Auto-detected if omitted.",
)
@click.option(
    "--default-remote",
    "default_remote",
    type=str,
    required=False,
    help="Default remote to use when the hostname doesn't specify one and ~/.trussrc has multiple remotes.",
)
@common.common_options()
def ssh_setup(python_path: Optional[str], default_remote: Optional[str]):
    """One-time setup: configure SSH access for Baseten workloads.

    Generates an SSH keypair, installs a ProxyCommand script, and adds a
    wildcard Host entry to ~/.ssh/config. After running this once, connect
    to any running workload with:

        # Training job
        ssh training-job-<job_id>-<node>.ssh.baseten.co

        # Model deployment (requires runtime.remote_ssh.enabled)
        ssh model-<model_id>-<deployment_id>.ssh.baseten.co
    """
    if default_remote is None:
        available_remotes = RemoteFactory.get_available_config_names()
        if len(available_remotes) == 1:
            default_remote = available_remotes[0]
        elif len(available_remotes) > 1:
            if check_is_interactive():
                default_remote = inquirer.select(
                    "Multiple remotes found. Which should be the default for SSH?",
                    qmark="",
                    choices=available_remotes,
                ).execute()
            else:
                console.print(
                    "[yellow]Multiple remotes found in ~/.trussrc. "
                    "Pass --default-remote to set one.[/yellow]"
                )

    key_path, reused = ensure_ssh_keypair()
    if reused:
        console.print(
            f"[yellow]WARNING: Existing SSH keypair found at {key_path}, reusing it.[/yellow]"
        )
    else:
        console.print(f"SSH keypair: {key_path}", style="dim")

    proxy_script = install_proxy_command_script(default_remote=default_remote)
    console.print(f"Proxy script: {proxy_script}", style="dim")

    setup_ssh_config(key_path=key_path, python_override=python_path)
    console.print("SSH config updated: ~/.ssh/config", style="dim")

    if default_remote:
        console.print(f"Default remote: {default_remote}", style="dim")

    console.print(
        "\n[green]SSH access configured.[/green] Connect to a running workload with:\n\n"
        "  Training job: ssh [bold]training-job-<job-id>-<node>.ssh.baseten.co[/bold]\n"
        "  Inference model: ssh [bold]model-<model-id>-<deployment-id>.ssh.baseten.co[/bold]"
    )
