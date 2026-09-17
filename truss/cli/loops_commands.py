import json
import shlex
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import rich.table
import rich_click as click
import yaml
from rich.markup import escape

import truss.cli.train.core as train_cli
import truss.cli.train_commands as train_commands
from truss.cli import remote_cli
from truss.cli.cli import truss_cli
from truss.cli.logs import utils as cli_log_utils
from truss.cli.logs.loops_deployment_log_watcher import LoopsDeploymentLogWatcher
from truss.cli.logs.model_log_watcher import ModelDeploymentLogWatcher
from truss.cli.logs.training_log_watcher import TrainingLogWatcher
from truss.cli.loops_checkpoint_viewer import (
    resolve_most_recent_run_for_base_model,
    view_loops_checkpoint_list,
)
from truss.cli.train import checkpoint_viewer as checkpoint_mod
from truss.cli.train.deploy_checkpoints.deploy_checkpoints import (
    TRAINER_CHECKPOINT_TARGET,
)
from truss.cli.train.exec import (
    BASETEN_API_KEY_ENV_VAR,
    DEFAULT_EXEC_PROJECT_NAME,
    SUPPORTED_EXEC_ACCELERATORS,
    UvProject,
    build_exec_project,
    ensure_team_api_key_secret,
    get_project_type,
    parse_environment_variables,
    validate_secret_references,
    validate_workspace_root,
)
from truss.cli.utils import common
from truss.cli.utils.output import console, json_command
from truss.remote.baseten.remote import BasetenRemote
from truss.remote.remote_factory import RemoteFactory
from truss_train import public_api as train_public_api

# A Loops client is an orchestration process: it tokenizes locally, holds a large
# connection pool, and drives GPU workers that idle out if it stalls. These are the
# shape people run it at by hand today, rather than the platform's own defaults.
LOOPS_EXEC_CPU_COUNT = 16
LOOPS_EXEC_MEMORY = "64Gi"


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
# Dead sampler states (DeploymentStatusV1) — hold no live GPUs; standalone
# samplers in these states are hidden from the usage view unless --all.
_TERMINAL_SAMPLER_STATUSES = frozenset(
    {
        "INACTIVE",
        "FAILED",
        "DEPLOY_FAILED",
        "BUILD_FAILED",
        "BUILD_STOPPED",
        "UNHEALTHY",
        "DEACTIVATING",
    }
)
# The usage table collapses raw backend statuses into the same buckets the UI
# (truss loops view) shows: trainer is ACTIVE/INACTIVE (scaled-to-zero, stopped,
# and failed all read INACTIVE); sampler adds SCALED_TO_ZERO. These are the
# statuses that read as ACTIVE — serving or coming up; anything else is INACTIVE.
_TRAINER_ACTIVE_STATUSES = frozenset({"CREATED", "DEPLOYING", "RUNNING"})
_SAMPLER_ACTIVE_STATUSES = frozenset(
    {"ACTIVE", "WAKING_UP", "UPDATING", "BUILDING", "DEPLOYING", "LOADING_MODEL"}
)


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
    aggregates GPUs in use vs scaled to zero. Standalone samplers (no trainer)
    appear as sampler-only rows. By default the table lists only allocations
    holding live GPUs; idle (scaled-to-zero) and terminal ones are counted in the
    summary but hidden unless you pass --all.
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
    scope = "org" if not mine else None
    deployments = remote_provider.api.list_loops_deployments(scope=scope)

    # Standalone samplers (no trainer) aren't in the deployments list. Fetch them
    # and drop any already shown under a deployment (matched by id, across all
    # deployments before the owner filter) so nothing is double-counted.
    paired_sampler_ids = {
        deployment["sampler"]["id"]
        for deployment in deployments
        if deployment.get("sampler") and deployment["sampler"].get("id")
    }
    standalone = [
        sampler
        for sampler in remote_provider.api.list_loops_samplers(scope=scope)
        if sampler.get("id") not in paired_sampler_ids
    ]

    if user_email:
        deployments = [
            deployment
            for deployment in deployments
            if (deployment.get("user") or {}).get("email") == user_email
        ]
        standalone = [
            sampler
            for sampler in standalone
            if (sampler.get("user") or {}).get("email") == user_email
        ]

    all_rows = deployments + [
        _standalone_sampler_as_row(sampler) for sampler in standalone
    ]
    # The summary reports true org totals regardless of what the table shows.
    summary = _compute_usage_summary(all_rows)

    # Default view lists only allocations holding live GPUs; idle (scaled-to-zero)
    # and terminal/dead ones are summarized above but hidden unless --all. Keeps
    # the table short on orgs with lots of idle allocations.
    if show_all:
        display_rows = all_rows
        hidden_count = 0
    else:
        display_rows = [row for row in all_rows if _row_holds_live_gpus(row)]
        hidden_count = len(all_rows) - len(display_rows)

    if output_format == checkpoint_mod.OUTPUT_FORMAT_JSON:
        _render_loops_usage_json(display_rows)
        return

    if not all_rows:
        console.print("No Loops deployments or samplers.", style="yellow")
        return

    _render_loops_usage(
        display_rows, summary, show_owner=not mine, hidden_count=hidden_count
    )


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
    """[DEPRECATED] Use `truss loops usage` instead.

    Lists Loops samplers visible to the caller.
    """
    console.print(
        "[DEPRECATED] `truss loops samplers view` is deprecated; use `truss loops usage` instead.",
        style="yellow",
    )
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


# Trainer-status placeholder for a standalone sampler (one with no trainer half).
_NO_TRAINER_STATUS = "—"


def _standalone_sampler_as_row(sampler: Dict[str, Any]) -> Dict[str, Any]:
    """Wrap a standalone sampler (no trainer) in the deployment row shape so the
    table and summary render it uniformly: empty trainer half, sampler populated."""
    return {
        "id": sampler.get("id", ""),
        "active_run_id": None,
        "latest_run_id": None,
        "base_model": sampler.get("base_model", ""),
        "created_at": sampler.get("created_at") or "",
        "status": {"name": _NO_TRAINER_STATUS},
        "user": sampler.get("user"),
        "instance_type": None,
        "node_count": 1,
        "sampler": {
            "status": sampler.get("status"),
            "instance_type": sampler.get("instance_type"),
            "node_count": sampler.get("node_count") or 1,
        },
    }


def _compute_usage_summary(deployments: List[Dict[str, Any]]) -> Dict[str, int]:
    """Aggregate GPU capacity into trainer/sampler and in-use/scaled-to-zero.

    Terminal deployments and dead (deactivated/failed) samplers hold no live
    capacity and are skipped entirely.
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
        sampler_status = (sampler.get("status") or {}).get("name")
        # A dead sampler (deactivated/failed) holds no live GPUs, even though it
        # still carries the instance type it last ran on — don't count it.
        if sampler_status in _TERMINAL_SAMPLER_STATUSES:
            continue
        sampler_capacity = _gpu_capacity(
            sampler_instance_type, sampler.get("node_count") or 1
        )
        if sampler_status == _SCALED_TO_ZERO_STATUS:
            summary["sampler_scaled_to_zero"] += sampler_capacity
        else:
            summary["sampler_in_use"] += sampler_capacity
    return summary


def _row_holds_live_gpus(row: Dict[str, Any]) -> bool:
    """Whether a row is actively holding GPUs (either half serving). Idle
    (scaled-to-zero) and terminal/dead allocations return False so the default
    view can hide them while the summary still counts them."""
    trainer_status = row["status"]["name"]
    trainer_live = (
        trainer_status not in _TERMINAL_DEPLOYMENT_STATUSES
        and trainer_status not in (_SCALED_TO_ZERO_STATUS, _NO_TRAINER_STATUS)
    )
    sampler = row.get("sampler")
    sampler_status = ((sampler or {}).get("status") or {}).get("name")
    sampler_live = (
        sampler is not None
        and sampler_status not in _TERMINAL_SAMPLER_STATUSES
        and (sampler_status != _SCALED_TO_ZERO_STATUS)
    )
    return trainer_live or sampler_live


def _trainer_status_label(status: str) -> str:
    """Collapse a raw trainer status to ACTIVE/INACTIVE (scaled-to-zero, stopped,
    and failed all read INACTIVE), or "—" for a standalone-sampler row."""
    if status == _NO_TRAINER_STATUS:
        return _NO_TRAINER_STATUS
    return "ACTIVE" if status in _TRAINER_ACTIVE_STATUSES else "INACTIVE"


def _sampler_status_label(status: Optional[str]) -> str:
    """Collapse a raw sampler status to ACTIVE/SCALED_TO_ZERO/INACTIVE, or "—"
    when there is no sampler."""
    if not status:
        return "—"
    if status == _SCALED_TO_ZERO_STATUS:
        return _SCALED_TO_ZERO_STATUS
    return "ACTIVE" if status in _SAMPLER_ACTIVE_STATUSES else "INACTIVE"


def _render_loops_usage(
    deployments: List[Dict[str, Any]],
    summary: Dict[str, int],
    show_owner: bool = False,
    hidden_count: int = 0,
) -> None:
    console.print(
        f"Trainer GPUs: {summary['trainer_in_use']} in use, "
        f"{summary['trainer_scaled_to_zero']} scaled to zero · "
        f"Sampler GPUs: {summary['sampler_in_use']} in use, "
        f"{summary['sampler_scaled_to_zero']} scaled to zero"
    )
    if hidden_count:
        plural = "s" if hidden_count != 1 else ""
        console.print(
            f"({hidden_count} idle or inactive allocation{plural} hidden — pass --all to show)",
            style="yellow",
        )
    if not deployments:
        return
    table = rich.table.Table(
        show_header=True,
        header_style="bold magenta",
        title="Loops GPU Usage",
        box=rich.table.box.ROUNDED,
        border_style="blue",
    )
    table.add_column("Latest Run", style="cyan")
    if show_owner:
        table.add_column("Owner", style="magenta")
    table.add_column("Base Model", style="green")
    table.add_column("Trainer GPU")
    table.add_column("Trainer Status")
    table.add_column("Sampler GPU")
    table.add_column("Sampler Status")
    table.add_column("Created")
    for deployment in deployments:
        sampler = deployment.get("sampler")
        created_at = deployment.get("created_at") or ""
        created = common.format_localized_time(created_at) if created_at else "—"
        sampler_status = _sampler_status_label(
            ((sampler or {}).get("status") or {}).get("name")
        )
        # Active-or-latest: idle (scaled-to-zero/stopped) deployments have no
        # active run but still expose their most recent run as a usable handle.
        run_id = deployment.get("active_run_id") or deployment.get("latest_run_id")
        row = [run_id or "—"]
        if show_owner:
            row.append((deployment.get("user") or {}).get("email") or "—")
        row.extend(
            [
                deployment.get("base_model", ""),
                _format_gpu_cell(
                    deployment.get("instance_type"), deployment.get("node_count") or 1
                ),
                _trainer_status_label(deployment["status"]["name"]),
                _format_gpu_cell(
                    sampler.get("instance_type") if sampler else None,
                    (sampler.get("node_count") or 1) if sampler else 1,
                ),
                sampler_status,
                created,
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
            "latest_run_id": deployment.get("latest_run_id"),
            "base_model": deployment.get("base_model", ""),
            "status": _trainer_status_label(deployment["status"]["name"]),
            "owner": (deployment.get("user") or {}).get("email"),
            "trainer_instance_type": deployment.get("instance_type"),
            "trainer_node_count": deployment.get("node_count"),
            "sampler_status": _sampler_status_label(
                (sampler.get("status") or {}).get("name")
            ),
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


def _print_exec_json(
    *,
    job_resp: dict,
    ssh_hostname: str,
    start_command: str,
    environment_variables: dict,
    cpu_count: int,
    memory: str,
    accelerator: Optional[str],
    gpu_count: int,
) -> None:
    """Emit the job's identity and shape as JSON on stdout.

    The job is not running yet when this is printed. Callers should poll
    `truss train view --job-id <id>` before connecting or expecting logs.
    """
    project = job_resp.get("training_project") or {}
    output = {
        "job_id": job_resp["id"],
        "project": {"id": project.get("id"), "name": project.get("name")},
        "ssh_hostname": ssh_hostname,
        "start_command": start_command,
        # Names only: an --env value can be as sensitive as a secret. Taken from the
        # built job rather than the flags, since the builder contributes its own.
        "environment_variables": sorted(environment_variables),
        "compute": {
            "cpu_count": cpu_count,
            "memory": memory,
            "accelerator": accelerator,
            "gpu_count": gpu_count if accelerator else None,
        },
        "job": job_resp,
    }
    # Flushed explicitly: with --tail the process keeps streaming logs after this,
    # and a block-buffered pipe would otherwise withhold the payload.
    print(json.dumps(output, indent=2), flush=True)


@loops.command(name="exec", context_settings={"ignore_unknown_options": True})
@click.argument("start_command", nargs=-1, type=click.UNPROCESSED)
@click.option(
    "--accelerator",
    type=click.Choice(SUPPORTED_EXEC_ACCELERATORS, case_sensitive=False),
    default=None,
    help="GPU accelerator type. Omit for a CPU-only job (the default).",
)
@click.option(
    "--gpu-count",
    type=click.IntRange(1, 8),
    default=None,
    help="Number of GPUs (1-8, default: 1). Requires --accelerator.",
)
@click.option(
    "--cpu-count",
    type=click.IntRange(min=1),
    default=LOOPS_EXEC_CPU_COUNT,
    show_default=True,
    help="Number of CPUs to request.",
)
@click.option(
    "--memory",
    type=str,
    default=LOOPS_EXEC_MEMORY,
    show_default=True,
    help="Memory to request (e.g. 8Gi).",
)
@click.option(
    "--project-name",
    type=str,
    required=False,
    help="Training project name (default: the name of the current directory).",
)
@click.option("--image", type=str, required=False, help="Custom Docker base image.")
@click.option(
    "--workspace-root",
    type=str,
    required=False,
    help=(
        "Directory to upload instead of just the current directory. Must be a "
        "parent of the current directory."
    ),
)
@click.option(
    "--exclude-dir",
    "exclude_dirs",
    type=str,
    multiple=True,
    help=(
        "Top-level directory of the workspace root to leave out of the upload. "
        "Repeatable."
    ),
)
@click.option(
    "--external-dir",
    "external_dirs",
    type=str,
    multiple=True,
    help="Directory outside the workspace root to include in the upload. Repeatable.",
)
@click.option(
    "--env",
    type=str,
    multiple=True,
    help="Environment variable for the job as KEY=VALUE. Repeatable.",
)
@click.option(
    "--secret",
    "secrets",
    type=str,
    multiple=True,
    help=(
        "Environment variable sourced from a Baseten workspace secret, as "
        "KEY=SECRET_NAME. Create secrets at https://app.baseten.co/settings/secrets. "
        "Repeatable."
    ),
)
@click.option(
    "-o",
    "--output-format",
    "output_format",
    type=click.Choice(["text", "json"], case_sensitive=False),
    default="text",
    help=(
        "Output format. 'json' emits structured JSON to stdout and all other "
        "output (progress, logs) to stderr."
    ),
)
@click.option(
    "--api-key/--no-api-key",
    "api_key",
    default=True,
    help=(
        "Set BASETEN_API_KEY in the job from a per-team secret, creating the key on "
        "first use. Pass --no-api-key for a job that should carry no credential."
    ),
)
@click.option(
    "--with-uv",
    is_flag=True,
    default=False,
    help=(
        "Make uv available in the job image. Your command should invoke uv itself, "
        "e.g. `truss loops exec --with-uv -- uv run python my_client.py`."
    ),
)
@click.option("--remote", type=str, required=False, help="Remote to use.")
@click.option(
    "--team",
    "provided_team_name",
    type=str,
    required=False,
    help="Team name for the training project",
)
@click.option(
    "--tail/--no-tail",
    default=False,
    show_default=True,
    help=(
        "Stream status + logs after push instead of returning immediately. With "
        "--tail, exec exits non-zero if the job fails."
    ),
)
@common.common_options()
@json_command
def exec_loops_command(
    start_command: tuple[str, ...],
    accelerator: Optional[str],
    gpu_count: Optional[int],
    cpu_count: int,
    memory: str,
    project_name: Optional[str],
    image: Optional[str],
    workspace_root: Optional[str],
    exclude_dirs: tuple[str, ...],
    external_dirs: tuple[str, ...],
    env: tuple[str, ...],
    secrets: tuple[str, ...],
    output_format: str,
    api_key: bool,
    with_uv: bool,
    remote: Optional[str],
    provided_team_name: Optional[str],
    tail: bool,
):
    """Run a Loops client from the current directory on Baseten.

    Archives the directory the command is invoked from, ships it to a training job
    sized for an orchestration client, and runs START_COMMAND there. Pass the
    command after `--`:

        truss loops exec -- python my_client.py

    START_COMMAND always runs last and verbatim. BASETEN_API_KEY is provided from a
    per-team secret unless you set it yourself; pass --with-uv to get uv in the job
    image. SSH into the job is available on demand.
    """
    as_json = output_format == "json"

    if not start_command:
        raise click.UsageError(
            "No start command given. Pass the command to run after `--`, "
            "e.g. `truss loops exec -- python my_client.py`."
        )
    if gpu_count is not None and accelerator is None:
        raise click.UsageError("--gpu-count requires --accelerator.")

    if accelerator:
        accelerator = accelerator.upper()
    gpu_count = gpu_count or 1

    environment_variables = parse_environment_variables(env=env, secrets=secrets)

    source_dir = Path.cwd()
    # Validate before any API call: truss_train's own check runs after the training
    # project has been created, which would leave a stray empty project behind.
    workspace_dir = validate_workspace_root(source_dir, workspace_root)
    if not project_name:
        # Repeated runs from the same checkout should group into one project.
        project_name = source_dir.name or DEFAULT_EXEC_PROJECT_NAME

    if not remote:
        remote = remote_cli.inquire_remote_name()

    remote_provider: BasetenRemote = cast(
        BasetenRemote, RemoteFactory.create(remote=remote)
    )
    if as_json:
        # The REST client prints 4xx messages straight to stdout, which would
        # corrupt the JSON stream. Let them surface as exceptions instead, so
        # json_command can render them as a structured error.
        remote_provider.api.suppress_error_print = True

    effective_team_name = provided_team_name or RemoteFactory.get_remote_team(remote)
    _, team_id = train_commands._resolve_team_name(
        remote_provider, effective_team_name, existing_project_name=project_name
    )
    validate_secret_references(
        remote_provider.api, environment_variables, team_id=team_id
    )

    # A training job is given no Baseten credential, so anything calling the Baseten
    # API needs one supplied. Only provision when the user hasn't named the variable
    # themselves, so an explicit --secret or --env always wins.
    if api_key and team_id and BASETEN_API_KEY_ENV_VAR not in environment_variables:
        # Deliberately not caught: an orphaned key needs the user's attention, and
        # continuing would hide it behind a successful push.
        api_key_secret = ensure_team_api_key_secret(remote_provider.api, team_id)
        if api_key_secret:
            environment_variables[BASETEN_API_KEY_ENV_VAR] = api_key_secret
            console.print(
                f"Using [cyan]{escape(BASETEN_API_KEY_ENV_VAR)}[/cyan] from the team "
                f"secret [cyan]{escape(api_key_secret.name)}[/cyan]."
            )
        else:
            console.print(
                f"Warning: could not provision a team Baseten API key, so "
                f"{BASETEN_API_KEY_ENV_VAR} will not be set in the job. Pass "
                f"`--secret {BASETEN_API_KEY_ENV_VAR}=<secret-name>` if the command "
                "calls the Baseten API.",
                style="yellow",
            )

    # --with-uv names uv explicitly, so it selects UvProject directly. Detection only
    # drives the warning below, and is the hook a future --project-type would use.
    detected_project = get_project_type(workspace_dir)
    training_project = build_exec_project(
        start_command=start_command,
        project_name=project_name,
        accelerator=accelerator,
        gpu_count=gpu_count,
        cpu_count=cpu_count,
        memory=memory,
        base_image=image,
        project=UvProject() if with_uv else None,
        workspace_root=workspace_root,
        exclude_dirs=exclude_dirs,
        external_dirs=external_dirs,
        environment_variables=environment_variables,
        enable_cache=True,
    )

    compute_str = (
        f"{gpu_count}x {accelerator}" if accelerator else f"{cpu_count} CPU / {memory}"
    )
    if not with_uv and detected_project is not None:
        console.print(
            f"Warning: this looks like a {detected_project.label} project, but "
            "--with-uv was not passed, so uv will not be present in the job image.",
            style="yellow",
        )

    # Escaped: `myproj[v2]` would otherwise be read as console markup, reporting a
    # different name than the one being pushed.
    console.print(
        f"Launching [cyan]{escape(project_name)}[/cyan] from "
        f"[cyan]{escape(str(source_dir))}[/cyan] on [cyan]{escape(compute_str)}[/cyan]..."
    )

    job_resp = train_public_api.push(
        config=training_project, remote=remote, source_dir=source_dir, team_id=team_id
    )

    job_id = job_resp["id"]
    project_id = job_resp["training_project"]["id"]
    ssh_hostname = f"training-job-{job_id}-0.ssh.baseten.co"

    if as_json:
        _print_exec_json(
            job_resp=job_resp,
            ssh_hostname=ssh_hostname,
            start_command=shlex.join(start_command),
            environment_variables=training_project.job.runtime.environment_variables,
            cpu_count=cpu_count,
            memory=memory,
            accelerator=accelerator,
            gpu_count=gpu_count,
        )
    else:
        console.print(
            f"\n[green]Job created![/green]\n"
            f"\n"
            f"SSH is available on demand. Check the interactive session with:\n"
            f"  [cyan]truss train isession --job-id {job_id}[/cyan]\n"
            f"\n"
            f"Then SSH in with:\n"
            f"  [cyan]ssh {ssh_hostname}[/cyan]\n"
            f"\n"
            f"If you haven't set up SSH yet, run:\n"
            f"  [cyan]truss ssh setup[/cyan]\n"
            f"\n"
            f"View logs:\n"
            f"  [cyan]truss train logs --job-id {job_id} --tail[/cyan]\n"
            f"\n"
            f"Stop the job:\n"
            f"  [cyan]truss train stop --job-id {job_id}[/cyan]"
        )

    if tail:
        watcher = TrainingLogWatcher(remote_provider.api, project_id, job_id)
        for log in watcher.watch():
            cli_log_utils.output_log(log)

        if watcher.failed:
            # Without this, `truss loops exec --tail -- pytest` is green in CI no
            # matter what the job did. sys.exit rather than click's Exit, which
            # subclasses RuntimeError and would be caught by `common_options`' error
            # handler and reported as "ERROR Exit: 1".
            sys.exit(1)
