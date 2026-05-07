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

    auth_header = remote_provider.fetch_auth_header()

    with console.status(
        # Trainer cold-start typically takes ~5 minutes.
        f"Deploying trainer and sampling servers for [cyan]{base_model}[/cyan]"
        f" (this may take ~5 minutes)...",
        spinner="dots",
    ):
        trainer_server = remote_provider.create_trainer_server(
            session_id=session_id, base_model=base_model
        )
        trainer_base_url = trainer_server["base_url"]
        sampler_base_url = (trainer_server.get("sampling_server") or {}).get("base_url")
        _poll_until_healthy(f"{trainer_base_url}/health", auth_header)
        if sampler_base_url:
            _poll_until_healthy(f"{sampler_base_url}/v1/models", auth_header)

    console.print(
        f"✨ Trainer deployment for [cyan]{base_model}[/cyan] is ready.\n"
        f"   Trainer server and sampling server have been provisioned.",
        style="green",
    )


@loops.command(name="deactivate")
@click.argument("base_model", type=str)
@click.option("--remote", type=str, required=False, help="Remote to use.")
@click.option(
    "--yes", "-y", is_flag=True, default=False, help="Skip confirmation prompt."
)
@common.common_options()
def deactivate_loop_deployment(
    base_model: str, remote: Optional[str], yes: bool
) -> None:
    """Deactivate the active loop deployment for BASE_MODEL.

    Shuts down the loop's deployment. Saved checkpoints remain accessible.
    """
    if not remote:
        remote = remote_cli.inquire_remote_name()

    remote_provider: BasetenRemote = cast(
        BasetenRemote, RemoteFactory.create(remote=remote)
    )

    if not yes:
        click.confirm(
            f"This will shut down the active loop deployment for {base_model}. Continue?",
            abort=True,
        )

    with console.status("Deactivating loop deployment...", spinner="dots"):
        remote_provider.deactivate_loop_deployment(base_model)

    console.print(f"Loop deployment for {base_model} deactivated.", style="green")


def _poll_until_healthy(health_url: str, auth_header: dict) -> None:
    """Poll health_url until it returns HTTP 200 or the timeout expires."""
    deadline = time.monotonic() + _READY_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        try:
            if requests.get(health_url, headers=auth_header, timeout=10).status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(_POLL_INTERVAL_SECONDS)
    raise click.ClickException(
        f"Timed out waiting for {health_url} to become healthy after"
        f" {_READY_TIMEOUT_SECONDS}s."
    )
