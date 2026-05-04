from typing import Optional

import rich_click as click
from InquirerPy import inquirer

from truss.cli.cli import truss_cli
from truss.cli.ssh import (
    ensure_ssh_keypair,
    install_proxy_command_script,
    setup_ssh_config,
)
from truss.cli.utils import common
from truss.cli.utils.common import check_is_interactive
from truss.cli.utils.output import console
from truss.remote.remote_factory import RemoteFactory


@click.group()
def ssh():
    """SSH access to Baseten workloads."""


truss_cli.add_command(ssh)


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
