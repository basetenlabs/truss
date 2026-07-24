import json
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
from truss.cli.train.deploy_checkpoints.deploy_checkpoints import (
    TRAINER_CHECKPOINT_TARGET,
)
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
@click.option(
    "--replicas",
    type=int,
    required=False,
    help=(
        "Number of data-parallel trainer replicas to provision. The trainer "
        "deployment runs this many copies of the model's preset node group "
        "(e.g. --replicas 4 on a 4-node preset → 16 nodes, 4 DP workers). "
        "Must be a positive integer; defaults to 1."
    ),
)
@click.option("--remote", type=str, required=False, help="Remote to use.")
@common.common_options()
def push_loops_deployment(
    base_model: str,
    project_id: Optional[str],
    replicas: Optional[int],
    remote: Optional[str],
) -> None:
    """Deploy a Loops run + sampler for a base model.

    Creates a Loops session, run, and paired sampler for BASE_MODEL. If
    the project already has an active Loops deployment for this base
    model, the command fails with a validation error.
    """
    if replicas is not None and replicas < 1:
        raise click.BadParameter(
            "--replicas must be a positive integer.", param_hint="--replicas"
        )

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
        remote_provider.create_loops_run(
            session_id=session_id, base_model=base_model, replicas=replicas
        )

    # Readiness is now the loops SDK's responsibility — clients block on
    # construction (TrainingClient → /health, SamplingClient → deployment
    # status). The CLI just confirms the resources were provisioned.
    console.print(
        f"✨ Loops deployment for [cyan]{base_model}[/cyan] provisioned.\n"
        f"   Trainer and sampler will finish coming up in the background",
        style="green",
    )


@loops.command(name="deactivate")
@click.argument("deployment_id", type=str, required=False)
@click.option("--run-id", type=str, required=False, help="Loops run ID to deactivate.")
@click.option("--remote", type=str, required=False, help="Remote to use.")
@click.option(
    "--yes", "-y", is_flag=True, default=False, help="Skip confirmation prompt."
)
@common.common_options()
def deactivate_loops_run(
    deployment_id: Optional[str],
    run_id: Optional[str],
    remote: Optional[str],
    yes: bool,
) -> None:
    """Deactivate a Loops run.

    Identify the run with --run-id. Shuts down the run, tearing down both of
    its halves (the run and its paired sampler). Saved checkpoints remain
    accessible. Use `truss loops view` to find run IDs.

    Passing a Loops deployment ID as a positional argument is deprecated;
    prefer --run-id.
    """
    if bool(run_id) == bool(deployment_id):
        raise click.UsageError("Pass exactly one of --run-id or a deployment ID.")

    if not remote:
        remote = remote_cli.inquire_remote_name()

    remote_provider: BasetenRemote = cast(
        BasetenRemote, RemoteFactory.create(remote=remote)
    )

    if deployment_id is not None:
        console.print(
            "[DEPRECATED] Passing a deployment ID is deprecated, use --run-id instead.",
            style="yellow",
        )
        if not yes:
            click.confirm(
                f"This will shut down Loops deployment {deployment_id}. Continue?",
                abort=True,
            )
        with console.status("Deactivating Loops deployment...", spinner="dots"):
            remote_provider.api.deactivate_loops_deployment(deployment_id)
        console.print(f"Loops deployment {deployment_id} deactivated.", style="green")
        return

    # Narrowed by the XOR check above: with no deployment_id, run_id is set.
    assert run_id is not None

    if not yes:
        click.confirm(f"This will shut down Loops run {run_id}. Continue?", abort=True)

    with console.status("Deactivating Loops run...", spinner="dots"):
        remote_provider.api.deactivate_loops_run(run_id)

    console.print(f"Loops run {run_id} deactivated.", style="green")


# INACTIVE runs are hidden by default (--all reveals them); ACTIVE is the only
# other run status.
_INACTIVE_RUN_STATUS = "INACTIVE"


@loops.command(name="view")
@click.option("--remote", type=str, required=False, help="Remote to use.")
@click.option(
    "--all", "show_all", is_flag=True, default=False, help="Include inactive runs."
)
@click.option(
    "--org",
    "org_wide",
    is_flag=True,
    default=False,
    help="List every Loops run in your organization (with its owner), not just your own.",
)
@click.option(
    "--reverse",
    "-r",
    is_flag=True,
    default=False,
    help="Reverse the default order (oldest first) so the most recent run is shown first.",
)
@click.option(
    "-o",
    "--output-format",
    type=click.Choice(
        [checkpoint_mod.OUTPUT_FORMAT_CLI_TABLE, checkpoint_mod.OUTPUT_FORMAT_JSON]
    ),
    default=checkpoint_mod.OUTPUT_FORMAT_CLI_TABLE,
    help="Output format: cli-table (default) or json.",
)
@common.common_options()
def view_loops_runs_summary(
    remote: Optional[str],
    show_all: bool,
    org_wide: bool,
    reverse: bool,
    output_format: str,
) -> None:
    """List Loops runs.

    Each row is a single run, keyed by its run ID, with a run-level status.
    Lists your own runs by default; pass --org to list every run in your
    organization, with an Owner column. Inactive runs are hidden unless you
    pass --all.
    """
    if not remote:
        remote = remote_cli.inquire_remote_name()

    remote_provider: BasetenRemote = cast(
        BasetenRemote, RemoteFactory.create(remote=remote)
    )
    runs = remote_provider.api.list_loops_runs(scope="org" if org_wide else None)
    runs = sorted(runs, key=lambda run: run.get("created_at") or "", reverse=reverse)
    is_human_output = output_format == checkpoint_mod.OUTPUT_FORMAT_CLI_TABLE

    if not runs and is_human_output:
        console.print("No Loops runs.", style="yellow")
        return

    if not show_all:
        runs = [run for run in runs if _run_status_name(run) != _INACTIVE_RUN_STATUS]
        if not runs and is_human_output:
            console.print(
                "No active Loops runs. Pass --all to include inactive runs.",
                style="yellow",
            )
            return

    if output_format == checkpoint_mod.OUTPUT_FORMAT_JSON:
        _render_loops_runs_summary_json(runs)
        return

    _render_loops_runs_summary(runs, show_owner=org_wide)


# Deployments in these terminal states are hidden from `usage` unless --all,
# and never counted toward GPU capacity.
_TERMINAL_DEPLOYMENT_STATUSES = frozenset({"STOPPED", "FAILED"})
# A deployment scaled to zero holds no live GPUs; it is tracked separately in
# the usage summary.
_SCALED_TO_ZERO_STATUS = "SCALED_TO_ZERO"


@loops.command(name="usage")
@click.option("--remote", type=str, required=False, help="Remote to use.")
@click.option(
    "--mine",
    is_flag=True,
    default=False,
    help="Show only your own Loops deployments (drops the Owner column).",
)
@click.option(
    "--user",
    "user_email",
    type=str,
    required=False,
    help="Filter to Loops deployments owned by this email (looked up org-wide).",
)
@click.option(
    "--all",
    "show_all",
    is_flag=True,
    default=False,
    help="Include deployments in terminal states (STOPPED, FAILED).",
)
@click.option(
    "-o",
    "--output-format",
    type=click.Choice(
        [checkpoint_mod.OUTPUT_FORMAT_CLI_TABLE, checkpoint_mod.OUTPUT_FORMAT_JSON]
    ),
    default=checkpoint_mod.OUTPUT_FORMAT_CLI_TABLE,
    help="Output format: cli-table (default) or json.",
)
@common.common_options()
def view_loops_usage(
    remote: Optional[str],
    mine: bool,
    user_email: Optional[str],
    show_all: bool,
    output_format: str,
) -> None:
    """Report Loops GPU capacity, one row per deployment (keyed by its run).

    Org-wide by default; pass --mine for just your own (which drops the Owner
    column) or --user to filter to a single owner. Each row shows the trainer
    and sampler GPU allocations and statuses, and a summary line above the table
    aggregates GPUs in use vs scaled to zero. Terminal deployments (STOPPED,
    FAILED) are hidden unless you pass --all.
    """
    if mine and user_email:
        raise click.UsageError("Pass at most one of --mine or --user.")

    if not remote:
        remote = remote_cli.inquire_remote_name()

    remote_provider: BasetenRemote = cast(
        BasetenRemote, RemoteFactory.create(remote=remote)
    )
    # --mine scopes the request to the caller; default and --user pull org-wide
    # and (for --user) filter to one owner client-side.
    deployments = remote_provider.api.list_loops_deployments(
        scope="org" if not mine else None
    )
    if user_email:
        deployments = [
            deployment
            for deployment in deployments
            if (deployment.get("user") or {}).get("email") == user_email
        ]
    is_human_output = output_format == checkpoint_mod.OUTPUT_FORMAT_CLI_TABLE

    if not deployments and is_human_output:
        console.print("No Loops deployments.", style="yellow")
        return

    if not show_all:
        deployments = [
            deployment
            for deployment in deployments
            if deployment["status"]["name"] not in _TERMINAL_DEPLOYMENT_STATUSES
        ]
        if not deployments and is_human_output:
            console.print(
                "No active Loops deployments. Pass --all to include "
                "STOPPED and FAILED deployments.",
                style="yellow",
            )
            return

    if output_format == checkpoint_mod.OUTPUT_FORMAT_JSON:
        _render_loops_usage_json(deployments)
        return

    _render_loops_usage(deployments, show_owner=not mine)


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
    """[DEPRECATED] Use `truss loops view` instead.

    Lists Loops runs visible to the caller. Both filters are optional and can be
    combined; omit both to list all runs visible to the caller.
    """
    console.print(
        "[DEPRECATED] `truss loops runs view` is deprecated; use `truss loops view` instead.",
        style="yellow",
    )
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


def _run_status_name(run: Dict[str, Any]) -> str:
    """The run's status name (ACTIVE/INACTIVE), from the server-defined status."""
    return (run.get("status") or {}).get("name") or ""


_RUN_STATUS_STYLES = {"ACTIVE": "green", "INACTIVE": "dim"}


def _render_loops_runs_summary(
    runs: List[Dict[str, Any]], show_owner: bool = False
) -> None:
    table = rich.table.Table(
        show_header=True,
        header_style="bold magenta",
        title="Loops Runs",
        box=rich.table.box.ROUNDED,
        border_style="blue",
    )
    table.add_column("Run ID", style="cyan")
    if show_owner:
        table.add_column("Owner", style="magenta")
    table.add_column("Base Model", style="green")
    table.add_column("Status")
    table.add_column("Created At")
    for run in runs:
        created_at = run.get("created_at") or ""
        created_str = common.format_localized_time(created_at) if created_at else ""
        status = _run_status_name(run)
        status_style = _RUN_STATUS_STYLES.get(status, "")
        row = [run.get("id", "")]
        if show_owner:
            row.append((run.get("user") or {}).get("email") or "—")
        row.extend(
            [
                run.get("base_model", ""),
                f"[{status_style}]{status}[/{status_style}]"
                if status_style
                else status,
                created_str,
            ]
        )
        table.add_row(*row)
    console.print(table)


def _render_loops_runs_summary_json(runs: List[Dict[str, Any]]) -> None:
    """Print the runs as jsonl. Closely follows the columns in the default format."""
    for run in runs:
        output = {
            "id": run.get("id", ""),
            "base_model": run.get("base_model", ""),
            "status": _run_status_name(run),
            "created_at": run.get("created_at") or "",
        }
        print(json.dumps(output))


def _format_gpu_cell(
    instance_type: Optional[Dict[str, Any]], node_count: int = 1
) -> str:
    """Render an instance type as "<gpu_type>:<gpu_count>", appending "×<nodes>"
    when the allocation spans more than one node; "—" when there is no GPU."""
    if not instance_type or not instance_type.get("gpu_count"):
        return "—"
    cell = f"{instance_type.get('gpu_type')}:{instance_type['gpu_count']}"
    if node_count and node_count > 1:
        cell += f" ×{node_count}"
    return cell


def _gpu_capacity(instance_type: Optional[Dict[str, Any]], node_count: int = 1) -> int:
    """Total GPUs an allocation holds: gpu_count across every node."""
    if not instance_type:
        return 0
    return (instance_type.get("gpu_count") or 0) * (node_count or 1)


def _compute_usage_summary(deployments: List[Dict[str, Any]]) -> Dict[str, int]:
    """Aggregate GPU capacity into trainer/sampler and in-use/scaled-to-zero.

    Terminal deployments hold no live capacity and are skipped entirely.
    """
    summary = {
        "trainer_in_use": 0,
        "trainer_scaled_to_zero": 0,
        "sampler_in_use": 0,
        "sampler_scaled_to_zero": 0,
    }
    for deployment in deployments:
        trainer_status = deployment["status"]["name"]
        if trainer_status in _TERMINAL_DEPLOYMENT_STATUSES:
            continue
        trainer_capacity = _gpu_capacity(
            deployment.get("instance_type"), deployment.get("node_count") or 1
        )
        if trainer_status == _SCALED_TO_ZERO_STATUS:
            summary["trainer_scaled_to_zero"] += trainer_capacity
        else:
            summary["trainer_in_use"] += trainer_capacity

        sampler = deployment.get("sampler")
        if not sampler:
            continue
        sampler_instance_type = sampler.get("instance_type")
        if not sampler_instance_type:
            continue
        sampler_capacity = _gpu_capacity(
            sampler_instance_type, sampler.get("node_count") or 1
        )
        sampler_status = (sampler.get("status") or {}).get("name")
        if sampler_status == _SCALED_TO_ZERO_STATUS:
            summary["sampler_scaled_to_zero"] += sampler_capacity
        else:
            summary["sampler_in_use"] += sampler_capacity
    return summary


def _render_loops_usage(
    deployments: List[Dict[str, Any]], show_owner: bool = False
) -> None:
    summary = _compute_usage_summary(deployments)
    console.print(
        f"Trainer GPUs: {summary['trainer_in_use']} in use, "
        f"{summary['trainer_scaled_to_zero']} scaled to zero · "
        f"Sampler GPUs: {summary['sampler_in_use']} in use, "
        f"{summary['sampler_scaled_to_zero']} scaled to zero"
    )
    table = rich.table.Table(
        show_header=True,
        header_style="bold magenta",
        title="Loops GPU Usage",
        box=rich.table.box.ROUNDED,
        border_style="blue",
    )
    table.add_column("Run ID", style="cyan")
    if show_owner:
        table.add_column("Owner", style="magenta")
    table.add_column("Base Model", style="green")
    table.add_column("Trainer GPU")
    table.add_column("Trainer Status")
    table.add_column("Sampler GPU")
    table.add_column("Sampler Status")
    table.add_column("Since")
    for deployment in deployments:
        sampler = deployment.get("sampler")
        created_at = deployment.get("created_at") or ""
        since = common.format_localized_time(created_at) if created_at else "—"
        sampler_status = ((sampler or {}).get("status") or {}).get("name") or "—"
        row = [deployment.get("active_run_id") or "—"]
        if show_owner:
            row.append((deployment.get("user") or {}).get("email") or "—")
        row.extend(
            [
                deployment.get("base_model", ""),
                _format_gpu_cell(
                    deployment.get("instance_type"), deployment.get("node_count") or 1
                ),
                deployment["status"]["name"],
                _format_gpu_cell(
                    sampler.get("instance_type") if sampler else None,
                    (sampler.get("node_count") or 1) if sampler else 1,
                ),
                sampler_status,
                since,
            ]
        )
        table.add_row(*row)
    console.print(table)


def _render_loops_usage_json(deployments: List[Dict[str, Any]]) -> None:
    """Print the deployments as jsonl. Mirrors the columns of the default table."""
    for deployment in deployments:
        sampler = deployment.get("sampler") or {}
        output = {
            "id": deployment.get("id", ""),
            "active_run_id": deployment.get("active_run_id"),
            "base_model": deployment.get("base_model", ""),
            "status": deployment["status"]["name"],
            "owner": (deployment.get("user") or {}).get("email"),
            "trainer_instance_type": deployment.get("instance_type"),
            "trainer_node_count": deployment.get("node_count"),
            "sampler_status": (sampler.get("status") or {}).get("name"),
            "sampler_instance_type": sampler.get("instance_type"),
            "sampler_node_count": sampler.get("node_count"),
            "created_at": deployment.get("created_at") or "",
        }
        print(json.dumps(output))


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
    # ID (the underlying model-deployment hashid).
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


def _parse_comma_separated(value: Optional[str]) -> List[str]:
    """Split a comma-separated option into a list, dropping empty entries."""
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _resolve_loops_checkpoint_names(
    remote_provider: BasetenRemote, run_id: str, names: List[str]
) -> List[str]:
    """Map Loops checkpoint names to their database IDs for the given run.

    Only deployable checkpoints (non-trainer targets) are considered, mirroring
    the interactive picker; trainer checkpoints hold training state and can't be
    served.
    """
    response = remote_provider.api.list_loops_checkpoints(run_id=run_id)
    deployable_checkpoints = [
        checkpoint
        for checkpoint in response.get("checkpoints", [])
        if checkpoint.get("target") != TRAINER_CHECKPOINT_TARGET
    ]
    name_to_id = {
        checkpoint["checkpoint_id"]: checkpoint["id"]
        for checkpoint in deployable_checkpoints
    }
    resolved_ids = []
    unknown_names = []
    for name in names:
        checkpoint_pk = name_to_id.get(name)
        if checkpoint_pk is None:
            unknown_names.append(name)
        else:
            resolved_ids.append(checkpoint_pk)
    if unknown_names:
        available = ", ".join(name_to_id) or "(none)"
        raise click.UsageError(
            f"Checkpoint name(s) {unknown_names} not found among deployable "
            f"checkpoints for Loops run {run_id}. Available: {available}."
        )
    return resolved_ids


@loops_checkpoints.command(name="deploy")
@click.option("--run-id", type=str, required=False, help="Loops run ID.")
@click.option(
    "--checkpoints",
    type=str,
    required=False,
    help="Comma-separated Loops checkpoint names (e.g. step-50,step-100). "
    "Requires --run-id, since names are scoped per run. Bypasses the "
    "interactive picker. Use `truss loops checkpoints view` to find names.",
)
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
    checkpoints: Optional[str],
    checkpoint_ids: Optional[str],
    config: Optional[str],
    dry_run: bool,
    remote: Optional[str],
) -> None:
    """Deploy checkpoints from a Loops run via vLLM.

    Identify checkpoints by name with --checkpoints (requires --run-id) or by
    database ID with --checkpoint-ids, or run without them to pick
    interactively.
    """
    if checkpoints and checkpoint_ids:
        raise click.UsageError(
            "--checkpoints and --checkpoint-ids are mutually exclusive. "
            "Deploy from one source: checkpoint names or checkpoint IDs."
        )
    if not run_id and not checkpoints and not checkpoint_ids and not config:
        raise click.UsageError(
            "Pass --run-id, --checkpoints, or --config (with "
            "loops_checkpoint_ids) to deploy Loops checkpoints."
        )
    if (checkpoints or checkpoint_ids) and config:
        raise click.UsageError(
            "--checkpoints / --checkpoint-ids cannot be combined with --config. "
            "Pick one source of checkpoint identifiers."
        )
    if checkpoints and not run_id:
        raise click.UsageError(
            "--checkpoints requires --run-id, since checkpoint names are scoped "
            "per run."
        )
    if checkpoint_ids and run_id:
        # Server resolves checkpoint PKs directly and ignores run_id when both
        # are present — silently dropping the run_id would mislead users into
        # thinking we validated their pairing.
        raise click.UsageError("--checkpoint-ids cannot be combined with --run-id.")

    parsed_checkpoint_names = _parse_comma_separated(checkpoints)
    if checkpoints and not parsed_checkpoint_names:
        raise click.UsageError(
            "--checkpoints parsed to an empty list. Provide one or more "
            "comma-separated Loops checkpoint names."
        )
    parsed_checkpoint_ids = _parse_comma_separated(checkpoint_ids)
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

    # Names are resolved to database IDs client-side, then feed the same
    # loops_checkpoint_ids path as --checkpoint-ids. The run_id is only used
    # for that resolution, so it is not forwarded downstream.
    if parsed_checkpoint_names:
        assert (
            run_id is not None
        )  # narrowed by the --checkpoints requires --run-id check
        parsed_checkpoint_ids = _resolve_loops_checkpoint_names(
            remote_provider, run_id, parsed_checkpoint_names
        )
        run_id = None

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

    The Loops deployments list returns each sampler with both ``deployment_id``
    (the OracleVersion id) and ``model_id`` (the Oracle id). Resolve the
    model_id client-side by matching on the deployment_id the caller gave us.
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


def _stream_loops_deployment_logs(
    remote_provider: BasetenRemote, loops_deployment_id: str, tail: bool
) -> None:
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


def _stream_model_deployment_logs(
    remote_provider: BasetenRemote, model_id: str, deployment_id: str, tail: bool
) -> None:
    if tail:
        model_watcher = ModelDeploymentLogWatcher(
            remote_provider.api, model_id, deployment_id
        )
        for log in model_watcher.watch():
            cli_log_utils.output_log(log)
    else:
        logs = remote_provider.api.get_model_deployment_logs(model_id, deployment_id)
        for log in cli_log_utils.parse_logs(logs):
            cli_log_utils.output_log(log)


@loops.command(name="logs")
@click.option(
    "--run-id", type=str, required=False, help="Loops run ID to fetch logs for."
)
@click.option(
    "--sampler",
    is_flag=True,
    default=False,
    help=(
        "With --run-id, tail the paired sampler's logs instead of the run's "
        "trainer logs. The two halves have separate log streams."
    ),
)
@click.option(
    "--loops-deployment-id",
    type=str,
    required=False,
    help=(
        "[DEPRECATED] Use --run-id to fetch the run's trainer logs; this will "
        "be removed in a future release."
    ),
)
@click.option(
    "--sampler-deployment-id",
    type=str,
    required=False,
    help=(
        "[DEPRECATED] Use --run-id --sampler instead; this will be removed in a "
        "future release. Fetch logs from the sampler's inference deployment by ID."
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
    run_id: Optional[str],
    sampler: bool,
    loops_deployment_id: Optional[str],
    sampler_deployment_id: Optional[str],
    tail: bool,
    remote: Optional[str],
) -> None:
    """Fetch logs for a Loops run.

    Identify the run with --run-id; by default this fetches the run's training
    logs, and --sampler fetches the paired sampler's logs instead. Use
    ``truss loops view`` to find run IDs.

    The deprecated --loops-deployment-id / --sampler-deployment-id flags fetch
    logs by deployment ID instead; prefer --run-id.
    """
    selectors = [run_id, loops_deployment_id, sampler_deployment_id]
    if sum(1 for selector in selectors if selector) != 1:
        raise click.UsageError(
            "Pass exactly one of --run-id, --loops-deployment-id, or "
            "--sampler-deployment-id."
        )
    if sampler and not run_id:
        raise click.UsageError("--sampler can only be used with --run-id.")

    if not remote:
        remote = remote_cli.inquire_remote_name()
    remote_provider: BasetenRemote = cast(
        BasetenRemote, RemoteFactory.create(remote=remote)
    )

    if loops_deployment_id is not None:
        console.print(
            "[DEPRECATED] --loops-deployment-id is deprecated, use --run-id instead.",
            style="yellow",
        )
        _stream_loops_deployment_logs(remote_provider, loops_deployment_id, tail)
        return

    if sampler_deployment_id is not None:
        console.print(
            "[DEPRECATED] --sampler-deployment-id is deprecated, use --run-id "
            "--sampler instead.",
            style="yellow",
        )
        model_id = _resolve_sampler_model_id(remote_provider, sampler_deployment_id)
        _stream_model_deployment_logs(
            remote_provider, model_id, sampler_deployment_id, tail
        )
        return

    # --run-id path: a single run object resolves both halves. ``deployment_id``
    # is the run's own deployment; the nested ``sampler`` carries the sampler's
    # inference deployment id plus its companion model id.
    # Narrowed by the one-selector check above.
    assert run_id is not None
    run = remote_provider.api.get_loops_run(run_id)

    if not sampler:
        run_deployment_id = run.get("deployment_id")
        if not run_deployment_id:
            raise click.ClickException(
                f"Loops run {run_id!r} has no trainer logs to fetch."
            )
        _stream_loops_deployment_logs(remote_provider, run_deployment_id, tail)
        return

    sampler_info = run.get("sampler") or {}
    resolved_sampler_deployment_id = sampler_info.get("deployment_id")
    resolved_model_id = sampler_info.get("model_id")
    if not resolved_sampler_deployment_id or not resolved_model_id:
        raise click.ClickException(
            f"Loops run {run_id!r} has no paired sampler to fetch logs from. "
            "Omit --sampler to view the run's trainer logs."
        )
    _stream_model_deployment_logs(
        remote_provider, resolved_model_id, resolved_sampler_deployment_id, tail
    )
