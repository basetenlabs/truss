import time
from typing import Any, Dict, List, Optional, cast

import requests
import rich.table
import rich_click as click
import yaml

import truss.cli.train.core as train_cli
from truss.cli import remote_cli
from truss.cli.cli import truss_cli
from truss.cli.loops_checkpoint_viewer import (
    resolve_most_recent_run_for_base_model,
    view_loops_checkpoint_list,
)
from truss.cli.train import checkpoint_viewer as checkpoint_mod
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


@loops.command(name="metrics")
@click.option(
    "--deployment-id",
    type=str,
    required=False,
    help="Loops deployment ID to fetch metrics for.",
)
@click.option(
    "--base-model",
    type=str,
    required=False,
    help=(
        "Base model name; resolves to the caller's active Loops deployment "
        "for that model. Errors if zero or more than one active deployment "
        "matches."
    ),
)
@click.option("--remote", type=str, required=False, help="Remote to use.")
@common.common_options()
def loops_metrics(
    deployment_id: Optional[str], base_model: Optional[str], remote: Optional[str]
) -> None:
    """Show utilization + request metrics for a Loops deployment.

    Pass exactly one of ``--deployment-id`` or ``--base-model``. Prints a
    single-snapshot view of GPU/CPU/memory utilization per pod plus
    service-level request metrics from the Knative queue-proxy.
    """
    if deployment_id and base_model:
        raise click.UsageError("Pass either --deployment-id or --base-model, not both.")
    if not deployment_id and not base_model:
        raise click.UsageError(
            "Pass --deployment-id or --base-model to identify a Loops deployment."
        )

    if not remote:
        remote = remote_cli.inquire_remote_name()
    remote_provider: BasetenRemote = cast(
        BasetenRemote, RemoteFactory.create(remote=remote)
    )

    resolved_deployment_id = deployment_id or _resolve_deployment_id_by_base_model(
        remote_provider, cast(str, base_model)
    )
    payload = remote_provider.api.get_loops_deployment_metrics(
        deployment_id=resolved_deployment_id
    )
    _render_loops_metrics(payload)


def _resolve_deployment_id_by_base_model(
    remote_provider: BasetenRemote, base_model: str
) -> str:
    deployments = remote_provider.api.list_loops_deployments()
    matches = [d for d in deployments if d.get("base_model") == base_model]
    if not matches:
        raise click.UsageError(
            f"No active Loops deployment found for base model {base_model!r}."
        )
    if len(matches) > 1:
        ids = ", ".join(d.get("id", "?") for d in matches)
        raise click.UsageError(
            f"Multiple active Loops deployments match {base_model!r}: {ids}. "
            "Pass --deployment-id to disambiguate."
        )
    return matches[0]["id"]


def _render_loops_metrics(payload: Dict[str, Any]) -> None:
    metrics = payload.get("metrics") or {}
    deployment_id = payload.get("deployment_id", "")

    console.print(
        f"[bold]Loops deployment metrics[/bold] for [cyan]{deployment_id}[/cyan]"
    )

    service_table = _build_service_metrics_table(metrics)
    console.print(service_table)

    per_node_table = _build_per_node_table(metrics.get("per_node_metrics") or [])
    if per_node_table is not None:
        console.print(per_node_table)
    else:
        console.print("No per-node compute metrics in this window.", style="yellow")


def _build_service_metrics_table(metrics: Dict[str, Any]) -> rich.table.Table:
    table = rich.table.Table(
        show_header=True,
        header_style="bold magenta",
        title="Service-level (latest)",
        box=rich.table.box.ROUNDED,
        border_style="blue",
    )
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    rate = _latest_value(metrics.get("inference_volume"))
    concurrent = _latest_value(metrics.get("concurrent_requests"))
    latencies = (metrics.get("response_time_stats") or [None])[-1] or {}

    table.add_row("Request rate (RPS)", _fmt_float(rate))
    table.add_row("Concurrent requests", _fmt_float(concurrent))
    table.add_row("Latency p50 (s)", _fmt_float(latencies.get("p50")))
    table.add_row("Latency p95 (s)", _fmt_float(latencies.get("p95")))
    table.add_row("Latency p99 (s)", _fmt_float(latencies.get("p99")))

    by_status = (metrics.get("inference_volume_by_status") or [None])[-1] or {}
    table.add_row("2xx rate", _fmt_float(by_status.get("status_2xx")))
    table.add_row("4xx rate", _fmt_float(by_status.get("status_4xx")))
    table.add_row("5xx rate", _fmt_float(by_status.get("status_5xx")))

    return table


def _build_per_node_table(nodes: List[Dict[str, Any]]) -> Optional[rich.table.Table]:
    if not nodes:
        return None
    table = rich.table.Table(
        show_header=True,
        header_style="bold magenta",
        title="Per-node compute (latest)",
        box=rich.table.box.ROUNDED,
        border_style="blue",
    )
    table.add_column("Node", style="cyan")
    table.add_column("CPU (cores)", justify="right")
    table.add_column("CPU mem", justify="right")
    table.add_column("GPU util", justify="right")
    table.add_column("GPU mem", justify="right")
    table.add_column("Eph. storage util", justify="right")

    for node in nodes:
        cpu = _fmt_float(_latest_value(node.get("cpu_usage")))
        cpu_mem = _fmt_bytes(_latest_value(node.get("cpu_memory_usage_bytes")))
        gpu_util = _fmt_gpu_map(node.get("gpu_utilization") or {}, as_percent=True)
        gpu_mem = _fmt_gpu_map(node.get("gpu_memory_usage_bytes") or {}, as_bytes=True)
        eph_storage = node.get("ephemeral_storage") or {}
        eph_util = _fmt_percent(_latest_value(eph_storage.get("utilization")))
        table.add_row(
            str(node.get("node_id", "")), cpu, cpu_mem, gpu_util, gpu_mem, eph_util
        )
    return table


def _latest_value(series: Optional[List[Dict[str, Any]]]) -> Optional[float]:
    if not series:
        return None
    last = series[-1]
    val = last.get("value")
    return float(val) if val is not None else None


def _fmt_float(value: Optional[float]) -> str:
    if value is None:
        return "—"
    return f"{value:.2f}"


def _fmt_percent(value: Optional[float]) -> str:
    if value is None:
        return "—"
    return f"{value * 100:.1f}%"


def _fmt_bytes(value: Optional[float]) -> str:
    if value is None:
        return "—"
    for unit, divisor in (("TB", 2**40), ("GB", 2**30), ("MB", 2**20), ("KB", 2**10)):
        if value >= divisor:
            return f"{value / divisor:.2f} {unit}"
    return f"{value:.0f} B"


def _fmt_gpu_map(
    per_gpu: Dict[str, List[Dict[str, Any]]],
    *,
    as_percent: bool = False,
    as_bytes: bool = False,
) -> str:
    """Latest value per GPU rank, joined as ``"0:val 1:val"``."""
    if not per_gpu:
        return "—"
    parts: List[str] = []
    for rank in sorted(per_gpu.keys(), key=lambda r: int(r) if r.isdigit() else r):
        latest = _latest_value(per_gpu[rank])
        if as_percent:
            parts.append(f"{rank}:{_fmt_percent(latest)}")
        elif as_bytes:
            parts.append(f"{rank}:{_fmt_bytes(latest)}")
        else:
            parts.append(f"{rank}:{_fmt_float(latest)}")
    return " ".join(parts)


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
