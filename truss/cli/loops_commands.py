from typing import Any, Dict, List, Optional, cast

import rich.table
import rich_click as click
import yaml

import truss.cli.train.core as train_cli
from truss.cli import remote_cli
from truss.cli.cli import truss_cli
from truss.cli.logs import utils as cli_log_utils
from truss.cli.logs.loops_deployment_log_watcher import LoopsDeploymentLogWatcher
from truss.cli.logs.model_log_watcher import ModelDeploymentLogWatcher
from truss.cli.loops_checkpoint_viewer import (
    resolve_most_recent_run_for_base_model,
    view_loops_checkpoint_list,
)
from truss.cli.train import checkpoint_viewer as checkpoint_mod
from truss.cli.utils import common
from truss.cli.utils.output import console
from truss.remote.baseten.remote import BasetenRemote
from truss.remote.remote_factory import RemoteFactory


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
        f"Provisioning Loops run and sampler for [cyan]{base_model}[/cyan]...",
        spinner="dots",
    ):
        remote_provider.create_loops_run(session_id=session_id, base_model=base_model)

    # Readiness is now the loops SDK's responsibility — clients block on
    # construction (TrainingClient → /health, SamplingClient → deployment
    # status). The CLI just confirms the resources were provisioned.
    console.print(
        f"✨ Loops deployment for [cyan]{base_model}[/cyan] provisioned.\n"
        f"   Trainer and sampler will finish coming up in the background",
        style="green",
    )


@loops.command(name="deactivate")
@click.argument("deployment_id", type=str)
@click.option("--remote", type=str, required=False, help="Remote to use.")
@click.option(
    "--yes", "-y", is_flag=True, default=False, help="Skip confirmation prompt."
)
@common.common_options()
def deactivate_loops_deployment(
    deployment_id: str, remote: Optional[str], yes: bool
) -> None:
    """Deactivate the Loops deployment with DEPLOYMENT_ID.

    Shuts down the Loops deployment. Saved checkpoints remain accessible.
    Use `truss loops view` to find deployment IDs.
    """
    if not remote:
        remote = remote_cli.inquire_remote_name()

    remote_provider: BasetenRemote = cast(
        BasetenRemote, RemoteFactory.create(remote=remote)
    )

    if not yes:
        click.confirm(
            f"This will shut down Loops deployment {deployment_id}. Continue?",
            abort=True,
        )

    with console.status("Deactivating Loops deployment...", spinner="dots"):
        remote_provider.api.deactivate_loops_deployment(deployment_id)

    console.print(f"Loops deployment {deployment_id} deactivated.", style="green")


_TERMINAL_DEPLOYMENT_STATUSES = frozenset({"STOPPED", "FAILED"})


@loops.command(name="view")
@click.option("--remote", type=str, required=False, help="Remote to use.")
@click.option(
    "--all",
    "show_all",
    is_flag=True,
    default=False,
    help="Include deployments in terminal states (STOPPED, FAILED).",
)
@common.common_options()
def view_loops_deployments(remote: Optional[str], show_all: bool) -> None:
    """List the caller's Loops deployments.

    By default, deployments in terminal states (STOPPED, FAILED) are hidden.
    Pass --all to include them.
    """
    if not remote:
        remote = remote_cli.inquire_remote_name()

    remote_provider: BasetenRemote = cast(
        BasetenRemote, RemoteFactory.create(remote=remote)
    )
    deployments = remote_provider.api.list_loops_deployments()
    if not deployments:
        console.print("No Loops deployments.", style="yellow")
        return
    if not show_all:
        deployments = [
            d
            for d in deployments
            if d["status"]["name"] not in _TERMINAL_DEPLOYMENT_STATUSES
        ]
        if not deployments:
            console.print(
                "No active Loops deployments. Pass --all to include "
                "STOPPED and FAILED deployments.",
                style="yellow",
            )
            return
    _render_loops_deployments(deployments)


@loops.group(name="runs")
def loops_runs() -> None:
    """Subcommands for working with Loops runs."""


@loops_runs.command(name="view")
@click.option("--run-id", type=str, required=False, help="Filter by run ID.")
@click.option(
    "--base-model", type=str, required=False, help="Filter runs by base model name."
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
    base_model: Optional[str],
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
    runs = remote_provider.api.list_loops_runs(run_id=run_id, base_model=base_model)
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
    table = rich.table.Table(
        show_header=True,
        header_style="bold magenta",
        title="Loops Deployments",
        box=rich.table.box.ROUNDED,
        border_style="blue",
    )
    table.add_column("Deployment ID", style="cyan")
    table.add_column("Base Model", style="green")
    table.add_column("Deployment Status")
    table.add_column("Deployment Base URL", style="blue")
    table.add_column("Sampler ID", style="cyan")
    table.add_column("Sampler Status")
    table.add_column("Sampler Base URL", style="blue")
    for deployment in deployments:
        sampler = deployment["sampler"]
        table.add_row(
            deployment["id"],
            deployment["base_model"],
            deployment["status"]["name"],
            deployment["base_url"],
            sampler["id"],
            sampler["status"]["name"],
            sampler["base_url"],
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
    # Two distinct IDs to surface: the user-facing Sampler ID (used by
    # ``truss loops samplers view --sampler-id``) and the Sampler Deployment
    # ID (the underlying model-deployment hashid, used by
    # ``truss loops logs --sampler-deployment-id``).
    table.add_column("Sampler ID", style="cyan")
    table.add_column("Sampler Deployment ID", style="cyan")
    table.add_column("Base Model", style="green")
    table.add_column("Base URL", style="blue")
    table.add_column("Created At")
    for sampler in samplers:
        created_at = sampler.get("created_at") or ""
        created_str = common.format_localized_time(created_at) if created_at else ""
        table.add_row(
            sampler.get("id", ""),
            sampler.get("deployment_id", "") or "",
            sampler.get("base_model", ""),
            sampler.get("base_url", ""),
            created_str,
        )
    console.print(table)


@loops.group(name="checkpoints")
def loops_checkpoints() -> None:
    """Subcommands for working with Loops checkpoints."""


@loops_checkpoints.command(name="view")
@click.option("--run-id", type=str, required=False, help="Loops run ID.")
@click.option(
    "--base-model",
    type=str,
    required=False,
    help="Base model name. Resolves to the most recent Loops run for that model.",
)
@click.option(
    "--sort",
    type=click.Choice(
        [
            checkpoint_mod.SORT_BY_CHECKPOINT_ID,
            checkpoint_mod.SORT_BY_SIZE,
            checkpoint_mod.SORT_BY_CREATED,
            checkpoint_mod.SORT_BY_TYPE,
        ]
    ),
    default=checkpoint_mod.SORT_BY_CREATED,
    help="Sort checkpoints by checkpoint-id, size, created date, or type.",
)
@click.option(
    "--order",
    type=click.Choice([checkpoint_mod.SORT_ORDER_ASC, checkpoint_mod.SORT_ORDER_DESC]),
    default=checkpoint_mod.SORT_ORDER_ASC,
    help="Sort order: ascending or descending.",
)
@click.option(
    "-o",
    "--output-format",
    type=click.Choice(
        [
            checkpoint_mod.OUTPUT_FORMAT_CLI_TABLE,
            checkpoint_mod.OUTPUT_FORMAT_CSV,
            checkpoint_mod.OUTPUT_FORMAT_JSON,
        ]
    ),
    default=checkpoint_mod.OUTPUT_FORMAT_CLI_TABLE,
    help="Output format: cli-table (default), csv, or json.",
)
@click.option("--remote", type=str, required=False, help="Remote to use.")
@common.common_options()
def view_loops_checkpoints(
    run_id: Optional[str],
    base_model: Optional[str],
    sort: str,
    order: str,
    output_format: str,
    remote: Optional[str],
) -> None:
    """List checkpoints for a Loops run.

    Identify the run with --run-id, or pass --base-model to pick the most
    recent run for that base model.
    """
    if run_id and base_model:
        raise click.UsageError("Pass either --run-id or --base-model, not both.")
    if not run_id and not base_model:
        raise click.UsageError("Pass --run-id or --base-model to identify a Loops run.")

    if not remote:
        remote = remote_cli.inquire_remote_name()

    remote_provider: BasetenRemote = cast(
        BasetenRemote, RemoteFactory.create(remote=remote)
    )

    if run_id:
        resolved_run_id = run_id
    else:
        try:
            assert base_model is not None  # narrowed by the validation above
            resolved_run_id = resolve_most_recent_run_for_base_model(
                remote_provider, base_model
            )
        except ValueError as e:
            raise click.UsageError(str(e))

    view_loops_checkpoint_list(
        remote_provider=remote_provider,
        run_id=resolved_run_id,
        sort_by=sort,
        order=order,
        output_format=output_format,
    )


@loops_checkpoints.command(name="deploy")
@click.option("--run-id", type=str, required=False, help="Loops run ID.")
@click.option(
    "--checkpoint-ids",
    type=str,
    required=False,
    help="Comma-separated Loops checkpoint IDs (e.g. vL3pQrS8,wK4tUvW9). "
    "Bypasses the interactive picker. Use `truss loops checkpoints view` to find IDs.",
)
@click.option(
    "--config",
    type=str,
    required=False,
    help="path to a python file that defines a DeployCheckpointsConfig",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Render the generated truss config to stdout without deploying.",
)
@click.option("--remote", type=str, required=False, help="Remote to use.")
@common.common_options()
def deploy_loops_checkpoints(
    run_id: Optional[str],
    checkpoint_ids: Optional[str],
    config: Optional[str],
    dry_run: bool,
    remote: Optional[str],
) -> None:
    """Deploy checkpoints from a Loops run via vLLM."""
    if not run_id and not checkpoint_ids and not config:
        raise click.UsageError(
            "Pass --run-id, --checkpoint-ids, or --config (with "
            "loops_checkpoint_ids) to deploy Loops checkpoints."
        )
    if checkpoint_ids and config:
        raise click.UsageError(
            "--checkpoint-ids cannot be combined with --config. "
            "Pick one source of checkpoint identifiers."
        )
    if checkpoint_ids and run_id:
        # Server resolves checkpoint PKs directly and ignores run_id when both
        # are present — silently dropping the run_id would mislead users into
        # thinking we validated their pairing.
        raise click.UsageError("--checkpoint-ids cannot be combined with --run-id.")

    parsed_checkpoint_ids = (
        [s.strip() for s in checkpoint_ids.split(",") if s.strip()]
        if checkpoint_ids
        else []
    )
    if checkpoint_ids and not parsed_checkpoint_ids:
        raise click.UsageError(
            "--checkpoint-ids parsed to an empty list. Provide one or more "
            "comma-separated Loops checkpoint IDs."
        )

    if not remote:
        remote = remote_cli.inquire_remote_name()

    remote_provider: BasetenRemote = cast(
        BasetenRemote, RemoteFactory.create(remote=remote)
    )

    result = train_cli.create_model_version_from_inference_template(
        remote_provider,
        train_cli.DeployCheckpointArgs(
            project_id=None,
            job_id=None,
            run_id=run_id,
            deploy_config_path=config,
            dry_run=dry_run,
            is_loops_command=True,
            checkpoint_ids=parsed_checkpoint_ids,
        ),
    )

    if dry_run:
        console.print("did not deploy because --dry-run flag provided", style="yellow")
        if result.truss_config:
            # Render to stdout so the user can pipe / inspect without
            # us littering the filesystem with truss_configs/ folders.
            print(yaml.safe_dump(result.truss_config.to_dict()))
    else:
        train_cli.print_deploy_checkpoints_success_message(result.deploy_config)


def _resolve_sampler_model_id(
    remote_provider: BasetenRemote, sampler_deployment_id: str
) -> str:
    """Find the model_id for a sampler's inference deployment.

    The Loops deployments list endpoint returns each deployment's sampler
    with both ``deployment_id`` (the OracleVersion id) and ``model_id`` (the
    Oracle id). We don't want users to have to pass both flags, so resolve
    the model_id client-side by matching on the deployment_id they gave us.
    """
    deployments = remote_provider.api.list_loops_deployments()
    for deployment in deployments:
        sampler = deployment.get("sampler") or {}
        if sampler.get("deployment_id") == sampler_deployment_id:
            return sampler["model_id"]
    raise click.ClickException(
        f"No Loops deployment found whose sampler matches deployment {sampler_deployment_id!r}. "
        "Run `truss loops view` to list active deployments."
    )


@loops.command(name="logs")
@click.option(
    "--loops-deployment-id",
    type=str,
    required=False,
    help=(
        "Fetch logs from a Loops deployment. The id is the "
        "``Deployment ID`` column in ``truss loops view``."
    ),
)
@click.option(
    "--sampler-deployment-id",
    type=str,
    required=False,
    help=(
        "Fetch logs from the sampler's inference deployment. The id is the "
        "``Sampler Deployment ID`` column in ``truss loops samplers view``. "
        "The companion model id is resolved automatically by matching "
        "against the caller's active Loops deployments."
    ),
)
@click.option(
    "--tail",
    is_flag=True,
    default=False,
    help="Continue polling for new log lines until the deployment goes inactive (or Ctrl+C).",
)
@click.option("--remote", type=str, required=False, help="Remote to use.")
@common.common_options()
def view_loops_logs(
    loops_deployment_id: Optional[str],
    sampler_deployment_id: Optional[str],
    tail: bool,
    remote: Optional[str],
) -> None:
    """Fetch logs from one half of a Loops deployment.

    Pass exactly one of ``--loops-deployment-id`` (the Loops deployment) or
    ``--sampler-deployment-id`` (the sampler's inference deployment). The
    two sides have separate log streams; pick the one you're debugging.
    """
    if bool(loops_deployment_id) == bool(sampler_deployment_id):
        raise click.UsageError(
            "Pass exactly one of --loops-deployment-id or --sampler-deployment-id."
        )

    if not remote:
        remote = remote_cli.inquire_remote_name()
    remote_provider: BasetenRemote = cast(
        BasetenRemote, RemoteFactory.create(remote=remote)
    )

    if loops_deployment_id is not None:
        if tail:
            loops_watcher = LoopsDeploymentLogWatcher(
                remote_provider.api, loops_deployment_id
            )
            for log in loops_watcher.watch():
                cli_log_utils.output_log(log)
        else:
            logs = remote_provider.api.get_loops_deployment_logs(loops_deployment_id)
            for log in cli_log_utils.parse_logs(logs):
                cli_log_utils.output_log(log)
        return

    # --sampler-deployment-id path: reuse the existing model-deployment log machinery.
    assert sampler_deployment_id is not None  # narrowed by the XOR check above
    model_id = _resolve_sampler_model_id(remote_provider, sampler_deployment_id)
    if tail:
        model_watcher = ModelDeploymentLogWatcher(
            remote_provider.api, model_id, sampler_deployment_id
        )
        for log in model_watcher.watch():
            cli_log_utils.output_log(log)
    else:
        logs = remote_provider.api.get_model_deployment_logs(
            model_id, sampler_deployment_id
        )
        for log in cli_log_utils.parse_logs(logs):
            cli_log_utils.output_log(log)
