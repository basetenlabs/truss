import time
from typing import Optional, cast

import rich_click as click

from truss.cli import remote_cli
from truss.cli.cli import truss_cli
from truss.cli.utils import common
from truss.cli.utils.output import console
from truss.remote.baseten.remote import BasetenRemote
from truss.remote.remote_factory import RemoteFactory

_POLL_INTERVAL_SECONDS = 10
_READY_TIMEOUT_SECONDS = (
    600  # ~10 minutes; trainer cold-start typically takes ~5 minutes
)


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
@click.option(
    "--sampler",
    type=str,
    required=False,
    help="Base model ID for additional samplers to include in the deployment.",
)
@click.option(
    "--max-seq-len",
    type=int,
    default=4096,
    show_default=True,
    help="Maximum sequence length for training.",
)
@click.option("--remote", type=str, required=False, help="Remote to use.")
@common.common_options()
def push_trainer_deployment(
    base_model: str,
    project_id: Optional[str],
    sampler: Optional[str],
    max_seq_len: int,
    remote: Optional[str],
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
        remote_provider.create_trainer_server(
            session_id=session_id, model=base_model, max_seq_len=max_seq_len
        )
        _poll_until_running(remote_provider, session_id)

    console.print(
        f"✨ Trainer deployment for [cyan]{base_model}[/cyan] is ready.\n"
        f"   Trainer server and sampling server have been provisioned.",
        style="green",
    )


def _poll_until_running(remote_provider: BasetenRemote, session_id: str) -> None:
    """Poll GET /v1/trainers/sessions/{session_id} until all trainer servers are RUNNING."""
    deadline = time.monotonic() + _READY_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        session_overview = remote_provider.get_trainer_session(session_id)
        trainer_servers = session_overview.get("trainer_servers", [])
        statuses = [ts.get("status") for ts in trainer_servers]
        if statuses and all(s == "running" for s in statuses):
            return
        if any(s == "failed" for s in statuses):
            raise click.ClickException(
                "Trainer deployment failed. Check the Baseten dashboard for details."
            )
        time.sleep(_POLL_INTERVAL_SECONDS)
    raise click.ClickException(
        f"Timed out waiting for trainer deployment to become ready after"
        f" {_READY_TIMEOUT_SECONDS}s."
    )
