"""CLI for managing persistent TrainerDeployments via the Baseten REST API.

These commands operate on long-lived trainer deployments that outlive any
single ``ServiceClient`` session — useful for keeping a warm GPU around
between iterative training runs instead of paying the 5–10 minute cold start
each time.
"""

from typing import Optional, cast

import rich_click as click

from truss.cli import remote_cli
from truss.cli.cli import truss_cli
from truss.cli.utils import common
from truss.cli.utils.output import console
from truss.remote.baseten.remote import BasetenRemote
from truss.remote.remote_factory import RemoteFactory


@click.group()
def trainers():
    """Subcommands for managing persistent trainer deployments."""


truss_cli.add_command(trainers)


def _resolve_remote(remote: Optional[str]) -> BasetenRemote:
    if not remote:
        remote = remote_cli.inquire_remote_name()
    return cast(BasetenRemote, RemoteFactory.create(remote=remote))


@trainers.command(name="deploy")
@click.argument("base_model", type=str)
@click.option(
    "--lora-rank",
    type=int,
    default=16,
    show_default=True,
    help="LoRA rank for the trainer.",
)
@click.option(
    "--max-seq-len",
    type=int,
    default=32768,
    show_default=True,
    help="Maximum sequence length for training.",
)
@click.option(
    "--scale-down-delay-seconds",
    type=int,
    default=3600,
    show_default=True,
    help="Seconds of inactivity before the trainer scales to zero.",
)
@click.option(
    "--team-id",
    type=str,
    required=False,
    help="Team to scope MDN weight mirroring to. Defaults to the caller's first team.",
)
@click.option(
    "--name",
    type=str,
    required=False,
    help="Optional human-readable name (unique per organization).",
)
@click.option("--remote", type=str, required=False, help="Remote to use.")
@common.common_options()
def deploy_trainer(
    base_model: str,
    lora_rank: int,
    max_seq_len: int,
    scale_down_delay_seconds: int,
    team_id: Optional[str],
    name: Optional[str],
    remote: Optional[str],
) -> None:
    """Deploy a persistent trainer for BASE_MODEL (e.g. 'Qwen/Qwen3-8B')."""
    provider = _resolve_remote(remote)
    deployment = provider.api.create_trainer_deployment(
        base_model=base_model,
        lora_rank=lora_rank,
        max_seq_len=max_seq_len,
        scale_down_delay_seconds=scale_down_delay_seconds,
        team_id=team_id,
        name=name,
    )
    console.print(
        f"✨ Trainer deployment created: [cyan]{deployment['id']}[/cyan]",
        style="green",
    )
    console.print(f"   base_model: {deployment.get('base_model')}")
    if deployment.get("name"):
        console.print(f"   name: {deployment['name']}")
    console.print(f"   base_url: {deployment.get('base_url')}")
    console.print(
        f"   stop later via: [cyan]'truss trainers stop {deployment['id']}'[/cyan]"
    )


@trainers.command(name="stop")
@click.argument("deployment_id", type=str)
@click.option("--remote", type=str, required=False, help="Remote to use.")
@common.common_options()
def stop_trainer(deployment_id: str, remote: Optional[str]) -> None:
    """Tear down the operator workload for a trainer deployment.

    The deployment row and its checkpoints are preserved — re-activate later
    with ``truss trainers start <id>``.
    """
    provider = _resolve_remote(remote)
    provider.api.deactivate_trainer_deployment(deployment_id)
    console.print(
        f"🛑 Trainer deployment [cyan]{deployment_id}[/cyan] deactivated.",
        style="green",
    )


@trainers.command(name="start")
@click.argument("deployment_id", type=str)
@click.option("--remote", type=str, required=False, help="Remote to use.")
@common.common_options()
def start_trainer(deployment_id: str, remote: Optional[str]) -> None:
    """Activate (or redeploy) a previously stopped trainer deployment."""
    provider = _resolve_remote(remote)
    provider.api.activate_trainer_deployment(deployment_id)
    console.print(
        f"🟢 Trainer deployment [cyan]{deployment_id}[/cyan] activating.",
        style="green",
    )


@trainers.command(name="info")
@click.argument("deployment_id", type=str)
@click.option("--remote", type=str, required=False, help="Remote to use.")
@common.common_options()
def info_trainer(deployment_id: str, remote: Optional[str]) -> None:
    """Print details for a trainer deployment."""
    provider = _resolve_remote(remote)
    deployment = provider.api.get_trainer_deployment(deployment_id)
    for key in ("id", "name", "base_model", "is_active", "base_url"):
        if deployment.get(key) is not None:
            console.print(f"{key}: {deployment[key]}")
