import time
from typing import Optional, cast

import requests
import rich_click as click

from truss.cli import remote_cli
from truss.cli.cli import truss_cli
from truss.cli.utils import common
from truss.cli.utils.output import console
from truss.remote.baseten.remote import BasetenRemote
from truss.remote.remote_factory import RemoteFactory

_READY_TIMEOUT_SECONDS = 600
_POLL_INTERVAL_SECONDS = 10


@click.group()
def loops():
    """Subcommands for truss loops"""


truss_cli.add_command(loops)


@loops.command(name="push")
@click.argument("base_model", type=str)
@click.option(
    "--project-id",
    type=str,
    required=False,
    help="Training project ID to associate the deployment with.",
)
@click.option("--remote", type=str, required=False, help="Remote to use.")
@common.common_options()
def push_trainer_deployment(
    base_model: str, project_id: Optional[str], remote: Optional[str]
) -> None:
    """Deploy training infrastructure for a base model.

    Creates a trainer session, trainer server, and sampling server for
    BASE_MODEL. If a training project already has an active trainer
    deployment, this command will fail with a validation error.
    """
    if not remote:
        remote = remote_cli.inquire_remote_name()

    remote_provider: BasetenRemote = cast(
        BasetenRemote, RemoteFactory.create(remote=remote)
    )

    with console.status("Creating trainer session...", spinner="dots"):
        session = remote_provider.create_trainer_session(training_project_id=project_id)
    session_id = session["id"]

    with console.status(
        # Trainer cold-start typically takes ~5 minutes.
        f"Deploying trainer and sampling servers for [cyan]{base_model}[/cyan]"
        f" (this may take ~5 minutes)...",
        spinner="dots",
    ):
        trainer_server = remote_provider.create_trainer_server(
            session_id=session_id, model=base_model
        )
        trainer_base_url = trainer_server["base_url"]
        _poll_until_running(remote_provider, trainer_base_url)

    console.print(
        f"✨ Trainer deployment for [cyan]{base_model}[/cyan] is ready.\n"
        f"   Trainer server and sampling server have been provisioned.",
        style="green",
    )


@loops.command(name="deactivate")
@click.argument("model_name", type=str)
@click.option("--remote", type=str, required=False, help="Remote to use.")
@click.option(
    "--yes",
    "-y",
    is_flag=True,
    default=False,
    help="Skip confirmation prompt.",
)
@common.common_options()
def deactivate_loop_deployment(
    model_name: str,
    remote: Optional[str],
    yes: bool,
) -> None:
    """Deactivate the active loop deployment for MODEL_NAME.

    Shuts down the loop's deployment. Saved checkpoints remain accessible.
    """
    if not remote:
        remote = remote_cli.inquire_remote_name()

    remote_provider: BasetenRemote = cast(
        BasetenRemote, RemoteFactory.create(remote=remote)
    )

    if not yes:
        click.confirm(
            f"This will shut down the active loop deployment for {model_name}. Continue?",
            abort=True,
        )

    with console.status("Deactivating loop deployment...", spinner="dots"):
        remote_provider.deactivate_loop_deployment(model_name)

    console.print(f"Loop deployment for {model_name} deactivated.", style="green")


def _poll_until_running(remote_provider: BasetenRemote, trainer_base_url: str) -> None:
    """Poll GET {trainer_base_url}/health until the trainer server is up."""
    health_url = f"{trainer_base_url}/health"
    auth_header = remote_provider.fetch_auth_header()
    deadline = time.monotonic() + _READY_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        try:
            resp = requests.get(health_url, headers=auth_header, timeout=10)
            if resp.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(_POLL_INTERVAL_SECONDS)
    raise click.ClickException(
        f"Timed out waiting for trainer deployment to become ready after"
        f" {_READY_TIMEOUT_SECONDS}s."
    )
