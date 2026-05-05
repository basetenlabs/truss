from typing import Optional, cast

import rich_click as click

from truss.cli import remote_cli
from truss.cli.cli import truss_cli
from truss.cli.utils import common
from truss.cli.utils.output import console
from truss.remote.baseten.remote import BasetenRemote
from truss.remote.remote_factory import RemoteFactory


@click.group()
def loops():
    """Subcommands for truss loops"""


truss_cli.add_command(loops)


@loops.command(name="push")
@click.argument("base_model_id", type=str)
@click.option(
    "--training-project-id",
    type=str,
    required=False,
    help="Training project ID to associate the deployment with.",
)
@click.option(
    "--sampler-checkpoint",
    type=str,
    required=False,
    help="Checkpoint ID to use for the sampling server.",
)
@click.option(
    "--trainer-checkpoint",
    type=str,
    required=False,
    help="Checkpoint ID to use for the trainer server.",
)
@click.option("--remote", type=str, required=False, help="Remote to use.")
@common.common_options()
def push_trainer_deployment(
    base_model_id: str,
    training_project_id: Optional[str],
    sampler_checkpoint: Optional[str],
    trainer_checkpoint: Optional[str],
    remote: Optional[str],
) -> None:
    """Deploy training infrastructure for a base model.

    Creates a trainer session, trainer server, and sampling server for
    BASE_MODEL_ID. If a training project already has an active trainer
    deployment, this command will fail with a validation error.
    """
    if not remote:
        remote = remote_cli.inquire_remote_name()

    remote_provider: BasetenRemote = cast(
        BasetenRemote, RemoteFactory.create(remote=remote)
    )

    with console.status("Creating trainer session...", spinner="dots"):
        session = remote_provider.create_trainer_session(
            training_project_id=training_project_id
        )
    session_id = session["id"]

    with console.status(
        f"Deploying trainer and sampling servers for [cyan]{base_model_id}[/cyan]...",
        spinner="dots",
    ):
        remote_provider.create_trainer_server(
            session_id=session_id,
            model=base_model_id,
            sampler_checkpoint_id=sampler_checkpoint,
            trainer_checkpoint_id=trainer_checkpoint,
        )

    console.print(
        f"✨ Trainer deployment for [cyan]{base_model_id}[/cyan] is ready.\n"
        f"   Trainer server and sampling server have been provisioned.",
        style="green",
    )
