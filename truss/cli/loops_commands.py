import time
from typing import Any, Dict, List, Optional, cast

import requests
import rich.table
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
def push_loops_deployment(
    base_model: str, project_id: Optional[str], remote: Optional[str]
) -> None:
    """Deploy a Loops run + sampler for a base model.

    Creates a Loops session, run, and paired sampler for BASE_MODEL. If
    the project already has an active Loops deployment for this base
    model, the command fails with a validation error.
    """
    if not remote:
        remote = remote_cli.inquire_remote_name()

    remote_provider: BasetenRemote = cast(
        BasetenRemote, RemoteFactory.create(remote=remote)
    )

    with console.status("Creating Loops session...", spinner="dots"):
        session = remote_provider.create_loops_session(training_project_id=project_id)
    session_id = session["id"]

    with console.status(
        # Loops run cold-start typically takes ~5 minutes.
        f"Deploying Loops run and sampler for [cyan]{base_model}[/cyan]"
        f" (this may take ~5 minutes)...",
        spinner="dots",
    ):
        run = remote_provider.create_loops_run(
            session_id=session_id, base_model=base_model
        )
        run_base_url = run["base_url"]
        _poll_until_running(remote_provider, run_base_url)

    console.print(
        f"✨ Loops deployment for [cyan]{base_model}[/cyan] is ready.\n"
        f"   Run and sampler have been provisioned.",
        style="green",
    )


@loops.command(name="deactivate")
@click.argument("base_model", type=str)
@click.option("--remote", type=str, required=False, help="Remote to use.")
@click.option(
    "--yes", "-y", is_flag=True, default=False, help="Skip confirmation prompt."
)
@common.common_options()
def deactivate_loops_deployment(
    base_model: str, remote: Optional[str], yes: bool
) -> None:
    """Deactivate the active Loops deployment for BASE_MODEL.

    Shuts down the Loops deployment. Saved checkpoints remain accessible.
    """
    if not remote:
        remote = remote_cli.inquire_remote_name()

    remote_provider: BasetenRemote = cast(
        BasetenRemote, RemoteFactory.create(remote=remote)
    )

    if not yes:
        click.confirm(
            f"This will shut down the active Loops deployment for {base_model}. Continue?",
            abort=True,
        )

    with console.status("Deactivating Loops deployment...", spinner="dots"):
        remote_provider.deactivate_loops_deployment(base_model)

    console.print(f"Loops deployment for {base_model} deactivated.", style="green")


def _poll_until_running(remote_provider: BasetenRemote, run_base_url: str) -> None:
    """Poll GET {run_base_url}/health until the Loops run is up."""
    health_url = f"{run_base_url}/health"
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
        f"Timed out waiting for Loops deployment to become ready after"
        f" {_READY_TIMEOUT_SECONDS}s."
    )


@loops.command(name="view")
@click.option("--remote", type=str, required=False, help="Remote to use.")
@common.common_options()
def view_loops_deployments(remote: Optional[str]) -> None:
    """List the caller's active Loops deployments.

    Excludes deployments whose latest status is STOPPED (filtered server-side).
    """
    if not remote:
        remote = remote_cli.inquire_remote_name()

    remote_provider: BasetenRemote = cast(
        BasetenRemote, RemoteFactory.create(remote=remote)
    )
    deployments = remote_provider.api.list_loops_deployments()
    _render_loops_deployments(deployments)


@loops.group(name="runs")
def loops_runs() -> None:
    """Subcommands for working with Loops runs."""


@loops_runs.command(name="view")
@click.option("--run-id", type=str, required=False, help="Filter by run ID.")
@click.option(
    "--model-name", type=str, required=False, help="Filter runs by base model name."
)
@click.option(
    "--reverse",
    "-r",
    is_flag=True,
    default=False,
    help="Reverse the default order (oldest first) so the most recent run is shown first.",
)
@click.option("--remote", type=str, required=False, help="Remote to use.")
@common.common_options()
def view_loops_runs(
    run_id: Optional[str],
    model_name: Optional[str],
    reverse: bool,
    remote: Optional[str],
) -> None:
    """List Loops runs visible to the caller.

    Both filters are optional and can be combined; omit both to list all
    runs visible to the caller.
    """
    if not remote:
        remote = remote_cli.inquire_remote_name()

    remote_provider: BasetenRemote = cast(
        BasetenRemote, RemoteFactory.create(remote=remote)
    )
    runs = remote_provider.api.list_loops_runs(run_id=run_id, base_model=model_name)
    runs = sorted(runs, key=lambda r: r.get("created_at") or "", reverse=reverse)
    _render_loops_runs(runs)


@loops.group(name="samplers")
def loops_samplers() -> None:
    """Subcommands for working with Loops samplers."""


@loops_samplers.command(name="view")
@click.option(
    "--reverse",
    "-r",
    is_flag=True,
    default=False,
    help="Reverse the default order (oldest first) so the most recent sampler is shown first.",
)
@click.option("--remote", type=str, required=False, help="Remote to use.")
@common.common_options()
def view_loops_samplers(reverse: bool, remote: Optional[str]) -> None:
    """List Loops samplers visible to the caller."""
    if not remote:
        remote = remote_cli.inquire_remote_name()

    remote_provider: BasetenRemote = cast(
        BasetenRemote, RemoteFactory.create(remote=remote)
    )
    samplers = remote_provider.api.list_loops_samplers()
    samplers = sorted(
        samplers, key=lambda s: s.get("created_at") or "", reverse=reverse
    )
    _render_loops_samplers(samplers)


def _render_loops_deployments(deployments: List[Dict[str, Any]]) -> None:
    if not deployments:
        console.print("No active Loops deployments.", style="yellow")
        return
    table = rich.table.Table(
        show_header=True,
        header_style="bold magenta",
        title="Loops Deployments",
        box=rich.table.box.ROUNDED,
        border_style="blue",
    )
    table.add_column("Deployment ID", style="cyan")
    table.add_column("Base Model", style="green")
    table.add_column("Base URL", style="blue")
    table.add_column("Deployment URL", style="blue")
    for deployment in deployments:
        sampler = deployment.get("sampler") or {}
        sampler_url = sampler.get("base_url", "") if isinstance(sampler, dict) else ""
        table.add_row(
            deployment.get("id", ""),
            deployment.get("base_model", "") or "",
            deployment.get("base_url", ""),
            sampler_url,
        )
    console.print(table)


def _render_loops_runs(runs: List[Dict[str, Any]]) -> None:
    if not runs:
        console.print("No Loops runs found.", style="yellow")
        return
    table = rich.table.Table(
        show_header=True,
        header_style="bold magenta",
        title="Loops Runs",
        box=rich.table.box.ROUNDED,
        border_style="blue",
    )
    table.add_column("Run ID", style="cyan")
    table.add_column("Session ID", style="cyan")
    table.add_column("Base Model", style="green")
    table.add_column("Base URL", style="blue")
    table.add_column("Created At")
    for run in runs:
        created_at = run.get("created_at") or ""
        created_str = common.format_localized_time(created_at) if created_at else ""
        table.add_row(
            run.get("id", ""),
            run.get("session_id", ""),
            run.get("base_model", ""),
            run.get("base_url", ""),
            created_str,
        )
    console.print(table)


def _render_loops_samplers(samplers: List[Dict[str, Any]]) -> None:
    if not samplers:
        console.print("No Loops samplers found.", style="yellow")
        return
    table = rich.table.Table(
        show_header=True,
        header_style="bold magenta",
        title="Loops Samplers",
        box=rich.table.box.ROUNDED,
        border_style="blue",
    )
    table.add_column("Sampler ID", style="cyan")
    table.add_column("Base Model", style="green")
    table.add_column("Base URL", style="blue")
    table.add_column("Created At")
    for sampler in samplers:
        created_at = sampler.get("created_at") or ""
        created_str = common.format_localized_time(created_at) if created_at else ""
        table.add_row(
            sampler.get("id", ""),
            sampler.get("base_model", ""),
            sampler.get("base_url", ""),
            created_str,
        )
    console.print(table)
