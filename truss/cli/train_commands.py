import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, cast

import rich_click as click
import yaml
from rich import box
from rich.table import Table

import truss.cli.train.core as train_cli
from truss.base.constants import TRAINING_TEMPLATE_DIR
from truss.cli import remote_cli
from truss.cli.cli import truss_cli
from truss.cli.logs import utils as cli_log_utils
from truss.cli.logs.training_log_watcher import TrainingLogWatcher
from truss.cli.resolvers.training_project_team_resolver import (
    resolve_training_project_team_name,
)
from truss.cli.train import common as train_common
from truss.cli.train import core
from truss.cli.train.cache import (
    OUTPUT_FORMAT_CLI_TABLE,
    OUTPUT_FORMAT_CSV,
    OUTPUT_FORMAT_JSON,
    SORT_BY_FILEPATH,
    SORT_BY_MODIFIED,
    SORT_BY_PERMISSIONS,
    SORT_BY_SIZE,
    SORT_BY_TYPE,
    SORT_ORDER_ASC,
    SORT_ORDER_DESC,
)
from truss.cli.train.types import DeploySuccessResult
from truss.cli.utils import common
from truss.cli.utils.output import console, error_console
from truss.remote.baseten.core import get_training_job_logs_with_pagination
from truss.remote.baseten.custom_types import TeamType
from truss.remote.baseten.remote import BasetenRemote
from truss.remote.remote_factory import RemoteFactory
from truss.util.path import copy_tree_path
from truss_train import Image, TrainingJob, TrainingProject

DEFAULT_LOCAL_WHETSTONE_PROJECT_NAME = "whetstone-local"


@click.group()
def train():
    """Subcommands for truss train"""


truss_cli.add_command(train)

_TEB_SUPPORTED_SKUS = {"h100", "h200", "b200"}


def _print_training_job_success_message(
    job_id: str,
    project_id: str,
    project_name: str,
    job_object: Optional[TrainingJob],
    remote_provider: BasetenRemote,
) -> None:
    """Print success message and helpful commands for a training job."""
    console.print("✨ Training job successfully created!", style="green")
    should_print_cache_summary = job_object and (
        job_object.runtime.enable_cache
        or job_object.runtime.cache_config
        and job_object.runtime.cache_config.enabled
    )
    cache_summary_snippet = ""
    if should_print_cache_summary:
        cache_summary_snippet = (
            f"📁 View cache summary via "
            f"[cyan]'truss train cache summarize \"{project_name}\"'[/cyan]\n"
        )
    console.print(
        f"🪵 View logs for your job via "
        f"[cyan]'truss train logs --job-id {job_id} --tail'[/cyan]\n"
        f"🔍 View metrics for your job via "
        f"[cyan]'truss train metrics --job-id {job_id}'[/cyan]\n"
        f"{cache_summary_snippet}"
        f"🌐 View job in the UI: {common.format_link(core.status_page_url(remote_provider.remote_url, project_id, job_id))}"
    )


def _handle_post_create_logic(
    job_resp: dict, remote_provider: BasetenRemote, tail: bool
) -> None:
    project_id, job_id = job_resp["training_project"]["id"], job_resp["id"]
    project_name = job_resp["training_project"]["name"]

    if job_resp.get("current_status", None) == "TRAINING_JOB_QUEUED":
        console.print(
            f"🟢 Training job is queued. You can check the status of your job by running 'truss train view --job-id={job_id}'.",
            style="green",
        )
    else:
        # recreate currently doesn't pass back a job object.
        _print_training_job_success_message(
            job_id,
            project_id,
            project_name,
            job_resp.get("job_object"),
            remote_provider,
        )

    if tail:
        watcher = TrainingLogWatcher(remote_provider.api, project_id, job_id)
        for log in watcher.watch():
            cli_log_utils.output_log(log)


def _prepare_click_context(f: click.Command, params: dict) -> click.Context:
    """create new click context for invoking a command via f.invoke(ctx)"""
    current_ctx = click.get_current_context()
    current_obj = current_ctx.find_root().obj

    ctx = click.Context(f, obj=current_obj)
    ctx.params = params
    return ctx


def _resolve_team_name(
    remote_provider: BasetenRemote,
    provided_team_name: Optional[str],
    existing_project_name: Optional[str] = None,
    existing_teams: Optional[dict[str, TeamType]] = None,
) -> tuple[Optional[str], Optional[str]]:
    return resolve_training_project_team_name(
        remote_provider=remote_provider,
        provided_team_name=provided_team_name,
        existing_project_name=existing_project_name,
        existing_teams=existing_teams,
    )


@train.command(name="push")
@click.argument("config", type=Path, required=True)
@click.option("--remote", type=str, required=False, help="Remote to use")
@click.option("--tail", is_flag=True, help="Tail for status + logs after push.")
@click.option("--job-name", type=str, required=False, help="Name of the training job.")
@click.option(
    "--team",
    "provided_team_name",
    type=str,
    required=False,
    help="Team name for the training project",
)
@common.common_options()
def push_training_job(
    config: Path,
    remote: Optional[str],
    tail: bool,
    job_name: Optional[str],
    provided_team_name: Optional[str],
):
    """Run a training job"""
    from truss_train import deployment, loader

    if not remote:
        remote = remote_cli.inquire_remote_name()

    remote_provider: BasetenRemote = cast(
        BasetenRemote, RemoteFactory.create(remote=remote)
    )

    existing_teams = remote_provider.api.get_teams()
    # Use config team as fallback if --team not provided
    effective_team_name = provided_team_name or RemoteFactory.get_remote_team(remote)

    with loader.import_training_project(config) as training_project:
        team_name, team_id = _resolve_team_name(
            remote_provider,
            effective_team_name,
            existing_project_name=training_project.name,
            existing_teams=existing_teams,
        )

        with console.status("Creating training job...", spinner="dots"):
            job_resp = deployment.create_training_job(
                remote_provider,
                config,
                training_project,
                job_name_from_cli=job_name,
                team_name=team_name,
                team_id=team_id,
            )

    # Note: This post create logic needs to happen outside the context
    # of the above context manager, as only one console session can be active
    # at a time.
    _handle_post_create_logic(job_resp, remote_provider, tail)


def _parse_accelerator_spec(accelerator: str) -> tuple[str, int, int]:
    parts = accelerator.split(":")
    accel_type = parts[0].strip()
    if not accel_type:
        raise click.UsageError("Invalid accelerator. Expected 'TYPE[:GPUS[:NODES]]'.")
    gpu_count = int(parts[1]) if len(parts) > 1 and parts[1] else 8
    node_count = int(parts[2]) if len(parts) > 2 and parts[2] else 1
    if gpu_count <= 0 or node_count <= 0:
        raise click.UsageError("Accelerator GPU count and node count must be > 0.")
    return accel_type, gpu_count, node_count


def _accelerator_supports_teb(accelerator_type: str) -> bool:
    return accelerator_type.strip().lower() in _TEB_SUPPORTED_SKUS


def _print_sft_summary(
    run_name: str,
    dataset: str,
    split: str,
    config_path: str,
    accelerator_type: str,
    target_gpus: int,
    final_yaml: str,
    suggestion: dict,
    output_path: Optional[Path] = None,
    show_yaml: bool = False,
) -> None:
    cfg = yaml.safe_load(final_yaml) or {}

    table = Table(
        title="Whetstone SFT Plan",
        box=box.ROUNDED,
        header_style="bold cyan",
        border_style="blue",
    )
    table.add_column("Section", style="bold")
    table.add_column("Key")
    table.add_column("Value", overflow="fold")

    model = cfg.get("model", "-")
    table.add_row("Run", "Name", str(run_name))
    table.add_row("Run", "Template", str(config_path))
    table.add_row("Data", "Dataset", str(dataset))
    table.add_row("Data", "Split", str(split))
    table.add_row("Model", "Model", str(model))
    table.add_row("Compute", "Accelerator", f"{accelerator_type} x {target_gpus} GPUs")
    table.add_row("Training", "LR", str(cfg.get("lr", "-")))
    table.add_row(
        "Training",
        "Epochs",
        str(cfg.get("num_train_epochs", cfg.get("max_epochs", "-"))),
    )
    table.add_row("Training", "Global Batch", str(cfg.get("global_batch_size", "-")))
    table.add_row("Training", "Micro Batch", str(cfg.get("micro_batch_size", "-")))
    table.add_row("Training", "Max Length", str(cfg.get("max_length", "-")))
    table.add_row(
        "Parallelism",
        "TP/PP/CP/EP",
        f"{cfg.get('tensor_model_parallel_size', '-')} / {cfg.get('pipeline_model_parallel_size', '-')} / {cfg.get('context_parallel_size', '-')} / {cfg.get('expert_model_parallel_size', '-')}",
    )
    planner_mode = "Skipped" if suggestion.get("teb_skipped") else "TEB Applied"
    table.add_row("Planner", "Decision", planner_mode)
    if suggestion.get("reason"):
        table.add_row("Planner", "Reason", str(suggestion.get("reason")))
    table.add_row("Planner", "Headroom (GiB)", str(suggestion.get("headroom_gb", "-")))
    table.add_row(
        "Planner",
        "Checkpoint Preset",
        str(
            suggestion.get("checkpoint_preset_applied")
            or suggestion.get("checkpoint_preset")
            or "-"
        ),
    )
    if output_path:
        table.add_row("Output", "Config File", str(output_path))

    console.print(table)
    if show_yaml:
        console.print("\n[bold]Rendered YAML[/bold]")
        console.print(final_yaml)


@train.command(name="sft")
@click.option("--remote", type=str, required=False, help="Remote to use")
@click.option("--dataset", type=str, required=True, help="Dataset ID.")
@click.option(
    "--config",
    type=str,
    default="configs/SFT/qwen3_4b.yaml",
    show_default=True,
    help="Whetstone config template path.",
)
@click.option(
    "--split", type=str, default="train", show_default=True, help="Dataset split."
)
@click.option(
    "--accelerator",
    type=str,
    default="H100:8:1",
    show_default=True,
    help="Accelerator in TYPE:GPUS:NODES format.",
)
@click.option("--name", type=str, required=False, help="Run name override.")
@click.option(
    "--max-seq-length", type=int, required=False, help="Override max sequence length."
)
@click.option(
    "--save-interval", type=int, required=False, help="Override save interval."
)
@click.option("--epochs", type=int, required=False, help="Override number of epochs.")
@click.option(
    "--learning-rate", type=str, required=False, help="Override learning rate."
)
@click.option(
    "--global-batch-size", type=int, required=False, help="Override global batch size."
)
@click.option(
    "--baseten",
    is_flag=True,
    help="Deprecated no-op. Non-dry-run now always creates a Whetstone training job.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Render and validate only. Save final config locally and do not create a job.",
)
@click.option(
    "--output-config",
    type=click.Path(dir_okay=False, writable=True, resolve_path=True),
    required=False,
    help="Optional output path for dry-run rendered config.",
)
@click.option("--show-yaml", is_flag=True, help="Print the full rendered YAML.")
@click.option("--tail", is_flag=True, help="Tail logs after creating job.")
@common.common_options()
def sft_training_job(
    remote: Optional[str],
    dataset: str,
    config: str,
    split: str,
    accelerator: str,
    name: Optional[str],
    max_seq_length: Optional[int],
    save_interval: Optional[int],
    epochs: Optional[int],
    learning_rate: Optional[str],
    global_batch_size: Optional[int],
    baseten: bool,
    dry_run: bool,
    output_config: Optional[str],
    show_yaml: bool,
    tail: bool,
):
    """Render + validate Whetstone SFT config via training API, optionally create a job."""
    if not remote:
        remote = remote_cli.inquire_remote_name()

    remote_provider: BasetenRemote = cast(
        BasetenRemote, RemoteFactory.create(remote=remote)
    )
    accel_type, gpu_count, node_count = _parse_accelerator_spec(accelerator)
    target_gpus = gpu_count * node_count

    config_request = {
        "config_path": config,
        "dataset": dataset,
        "run_name": name,
        "split": split,
        "max_seq_length": max_seq_length,
        "save_interval": save_interval,
        "max_epochs": epochs,
        "learning_rate": learning_rate,
        "global_batch_size": global_batch_size,
    }
    config_request = {k: v for k, v in config_request.items() if v is not None}

    with console.status("Rendering Whetstone config...", spinner="dots"):
        render_resp = remote_provider.api.render_whetstone_config(config_request)

    rendered_yaml = render_resp["rendered_yaml"]
    run_name = render_resp["run_name"]
    rendered_cfg = yaml.safe_load(rendered_yaml) or {}
    planner_gbs = global_batch_size or rendered_cfg.get("global_batch_size")
    if not planner_gbs:
        raise click.UsageError(
            "global_batch_size is required either via --global-batch-size or in template defaults."
        )

    if _accelerator_supports_teb(accel_type):
        with console.status("Running TEB validation...", spinner="dots"):
            plan_resp = remote_provider.api.validate_whetstone_plan(
                {
                    "rendered_yaml": rendered_yaml,
                    "accelerator": accel_type,
                    "target_gpus": target_gpus,
                    "global_batch_size": int(planner_gbs),
                    "apply_checkpointing": True,
                }
            )

        if not plan_resp.get("available", False):
            raise click.ClickException(
                f"TEB planner unavailable: {plan_resp.get('reason', 'unknown error')}"
            )
        if not plan_resp.get("found", False):
            raise click.ClickException(
                f"No feasible plan found: {plan_resp.get('reason', 'unknown reason')}"
            )

        final_yaml = plan_resp["rendered_yaml"]
        suggestion = plan_resp.get("suggestion", {})
    else:
        final_yaml = rendered_yaml
        suggestion = {
            "teb_skipped": True,
            "reason": f"unsupported hardware sku '{accel_type.lower()}'",
        }
        console.print(
            f"Skipping TEB validation for accelerator '{accel_type}' (unsupported by local TEB).",
            style="yellow",
        )

    if dry_run:
        output_path = (
            Path(output_config) if output_config else Path.cwd() / f"{run_name}.yaml"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(final_yaml, encoding="utf-8")
        _print_sft_summary(
            run_name=run_name,
            dataset=dataset,
            split=split,
            config_path=config,
            accelerator_type=accel_type,
            target_gpus=target_gpus,
            final_yaml=final_yaml,
            suggestion=suggestion,
            output_path=output_path,
            show_yaml=show_yaml,
        )
        console.print("✨ Dry run complete.", style="green")
        return

    _print_sft_summary(
        run_name=run_name,
        dataset=dataset,
        split=split,
        config_path=config,
        accelerator_type=accel_type,
        target_gpus=target_gpus,
        final_yaml=final_yaml,
        suggestion=suggestion,
        show_yaml=show_yaml,
    )

    resolved_project_id = _get_or_create_default_local_whetstone_project_id(
        remote_provider
    )

    create_req = {
        "config": config_request,
        "planner": {
            "accelerator": accel_type,
            "target_gpus": target_gpus,
            "global_batch_size": int(planner_gbs),
            "apply_checkpointing": True,
        },
        "runtime": {"name": run_name},
    }
    with console.status("Creating Whetstone training job...", spinner="dots"):
        create_resp = remote_provider.api.create_whetstone_training_job(
            resolved_project_id, create_req
        )
    _handle_post_create_logic(create_resp["training_job"], remote_provider, tail)


@train.command(name="recreate")
@click.option(
    "--job-id", type=str, required=False, help="Job ID of Training Job to recreate"
)
@click.option("--remote", type=str, required=False, help="Remote to use")
@click.option("--tail", is_flag=True, help="Tail for status + logs after recreation.")
@common.common_options()
def recreate_training_job(job_id: Optional[str], remote: Optional[str], tail: bool):
    """Recreate an existing training job from an existing job ID"""
    if not remote:
        remote = remote_cli.inquire_remote_name()

    remote_provider: BasetenRemote = cast(
        BasetenRemote, RemoteFactory.create(remote=remote)
    )

    console.print("Recreating training job...", style="bold")
    job_resp = train_cli.recreate_training_job(
        remote_provider=remote_provider, job_id=job_id
    )

    _handle_post_create_logic(job_resp, remote_provider, tail)


@train.command(name="logs")
@click.option("--remote", type=str, required=False, help="Remote to use")
@click.option("--project-id", type=str, required=False, help="Project ID.")
@click.option("--project", type=str, required=False, help="Project name or project id.")
@click.option("--job-id", type=str, required=False, help="Job ID.")
@click.option("--tail", is_flag=True, help="Tail for ongoing logs.")
@common.common_options()
def get_job_logs(
    remote: Optional[str],
    project_id: Optional[str],
    project: Optional[str],
    job_id: Optional[str],
    tail: bool,
):
    """Fetch logs for a training job"""

    if not remote:
        remote = remote_cli.inquire_remote_name()

    remote_provider: BasetenRemote = cast(
        BasetenRemote, RemoteFactory.create(remote=remote)
    )
    project_id = _maybe_resolve_project_id_from_id_or_name(
        remote_provider, project_id=project_id, project=project
    )

    project_id, job_id = train_common.get_most_recent_job(
        remote_provider, project_id, job_id
    )

    if not tail:
        logs = get_training_job_logs_with_pagination(
            remote_provider.api, project_id, job_id
        )
        for log in cli_log_utils.parse_logs(logs):
            cli_log_utils.output_log(log)
    else:
        log_watcher = TrainingLogWatcher(remote_provider.api, project_id, job_id)
        for log in log_watcher.watch():
            cli_log_utils.output_log(log)


@train.command(name="stop")
@click.option("--project-id", type=str, required=False, help="Project ID.")
@click.option("--project", type=str, required=False, help="Project name or project id.")
@click.option("--job-id", type=str, required=False, help="Job ID.")
@click.option("--all", is_flag=True, help="Stop all running jobs.")
@click.option("--remote", type=str, required=False, help="Remote to use")
@common.common_options()
def stop_job(
    project_id: Optional[str],
    project: Optional[str],
    job_id: Optional[str],
    all: bool,
    remote: Optional[str],
):
    """Stop a training job"""

    if not remote:
        remote = remote_cli.inquire_remote_name()

    remote_provider: BasetenRemote = cast(
        BasetenRemote, RemoteFactory.create(remote=remote)
    )
    project_id = _maybe_resolve_project_id_from_id_or_name(
        remote_provider, project_id=project_id, project=project
    )
    if all:
        train_cli.stop_all_jobs(remote_provider, project_id)
    else:
        project_id, job_id = train_cli.get_args_for_stop(
            remote_provider, project_id, job_id
        )
        remote_provider.api.stop_training_job(project_id, job_id)
        console.print("Training job stopped successfully.", style="green")


@train.command(name="view")
@click.option(
    "--project-id", type=str, required=False, help="View training jobs for a project."
)
@click.option("--project", type=str, required=False, help="Project name or project id.")
@click.option(
    "--job-id", type=str, required=False, help="View a specific training job."
)
@click.option("--remote", type=str, required=False, help="Remote to use")
@common.common_options()
def view_training(
    project_id: Optional[str],
    project: Optional[str],
    job_id: Optional[str],
    remote: Optional[str],
):
    """List all training jobs for a project"""

    if not remote:
        remote = remote_cli.inquire_remote_name()

    remote_provider: BasetenRemote = cast(
        BasetenRemote, RemoteFactory.create(remote=remote)
    )
    project_id = _maybe_resolve_project_id_from_id_or_name(
        remote_provider, project_id=project_id, project=project
    )

    train_cli.view_training_details(remote_provider, project_id, job_id)


@train.command(name="metrics")
@click.option("--project-id", type=str, required=False, help="Project ID.")
@click.option("--project", type=str, required=False, help="Project name or project id.")
@click.option("--job-id", type=str, required=False, help="Job ID.")
@click.option("--remote", type=str, required=False, help="Remote to use")
@common.common_options()
def get_job_metrics(
    project_id: Optional[str],
    project: Optional[str],
    job_id: Optional[str],
    remote: Optional[str],
):
    """Get metrics for a training job"""

    if not remote:
        remote = remote_cli.inquire_remote_name()

    remote_provider: BasetenRemote = cast(
        BasetenRemote, RemoteFactory.create(remote=remote)
    )
    project_id = _maybe_resolve_project_id_from_id_or_name(
        remote_provider, project_id=project_id, project=project
    )
    train_cli.view_training_job_metrics(remote_provider, project_id, job_id)


@train.command(name="deploy_checkpoints")
@click.option("--project-id", type=str, required=False, help="Project ID.")
@click.option("--project", type=str, required=False, help="Project name or project id.")
@click.option("--job-id", type=str, required=False, help="Job ID.")
@click.option(
    "--config",
    type=str,
    required=False,
    help="path to a python file that defines a DeployCheckpointsConfig",
)
@click.option(
    "--dry-run", is_flag=True, help="Generate a truss config without deploying"
)
@click.option(
    "--truss-config-output-dir",
    type=str,
    required=False,
    help="Path to output the truss config to. If not provided, will output to truss_configs/<model_version_name>_<model_version_id> or truss_configs/dry_run_<timestamp> if dry run.",
)
@click.option("--remote", type=str, required=False, help="Remote to use")
@common.common_options()
def deploy_checkpoints(
    project_id: Optional[str],
    project: Optional[str],
    job_id: Optional[str],
    config: Optional[str],
    remote: Optional[str],
    dry_run: bool,
    truss_config_output_dir: Optional[str],
):
    """
    Deploy a LoRA checkpoint via vLLM.
    """

    if not remote:
        remote = remote_cli.inquire_remote_name()

    remote_provider: BasetenRemote = cast(
        BasetenRemote, RemoteFactory.create(remote=remote)
    )
    project_id = _maybe_resolve_project_id_from_id_or_name(
        remote_provider, project_id=project_id, project=project
    )
    result = train_cli.create_model_version_from_inference_template(
        remote_provider,
        train_cli.DeployCheckpointArgs(
            project_id=project_id,
            job_id=job_id,
            deploy_config_path=config,
            dry_run=dry_run,
        ),
    )

    if dry_run:
        console.print("did not deploy because --dry-run flag provided", style="yellow")

    _write_truss_config(result, truss_config_output_dir, dry_run)

    if not dry_run:
        train_cli.print_deploy_checkpoints_success_message(result.deploy_config)


def _write_truss_config(
    result: DeploySuccessResult, truss_config_output_dir: Optional[str], dry_run: bool
) -> None:
    if not result.truss_config:
        return
    # format: 20251006_123456
    datestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = (
        f"{result.model_version.name}_{result.model_version.id}"
        if result.model_version
        else f"dry_run_{datestamp}"
    )
    output_dir_str = truss_config_output_dir or f"truss_configs/{folder_name}"
    output_dir = Path(output_dir_str)
    output_path = output_dir / "config.yaml"
    os.makedirs(output_dir, exist_ok=True)
    console.print(f"Writing truss config to {output_path}", style="yellow")
    console.print(f"👀 Run `cat {output_path}` to view the truss config", style="green")
    if dry_run:
        console.print(
            f"🚀 Run `cd {output_dir} && truss push --publish` to deploy the truss",
            style="green",
        )
    result.truss_config.write_to_yaml_file(output_path)


@train.command(name="download")
@click.option("--job-id", type=str, required=True, help="Job ID.")
@click.option("--remote", type=str, required=False, help="Remote to use")
@click.option(
    "--target-directory",
    type=click.Path(file_okay=False, dir_okay=True, writable=True, resolve_path=True),
    required=False,
    help="Directory where the file should be downloaded. Defaults to current directory.",
)
@click.option(
    "--no-unzip",
    is_flag=True,
    help="Instructs truss to not unzip the folder upon download.",
)
@common.common_options()
def download_training_job(
    job_id: str, remote: Optional[str], target_directory: Optional[str], no_unzip: bool
) -> None:
    if not job_id:
        error_console.print("Job ID is required")
        sys.exit(1)

    if not remote:
        remote = remote_cli.inquire_remote_name()

    remote_provider: BasetenRemote = cast(
        BasetenRemote, RemoteFactory.create(remote=remote)
    )

    try:
        with console.status(
            "[bold green]Downloading training job data...", spinner="dots"
        ):
            target_path = train_cli.download_training_job_data(
                remote_provider=remote_provider,
                job_id=job_id,
                target_directory=target_directory,
                unzip=not no_unzip,
            )

        console.print(
            f"✨ Training job data downloaded to {target_path}", style="bold green"
        )
    except Exception as e:
        error_console.print(f"Failed to download training job data: {str(e)}")
        sys.exit(1)


@train.command(name="get_checkpoint_urls")
@click.option("--job-id", type=str, required=False, help="Job ID.")
@click.option("--remote", type=str, required=False, help="Remote to use")
@common.common_options()
def download_checkpoint_artifacts(job_id: Optional[str], remote: Optional[str]) -> None:
    if not remote:
        remote = remote_cli.inquire_remote_name()

    remote_provider: BasetenRemote = cast(
        BasetenRemote, RemoteFactory.create(remote=remote)
    )

    try:
        with console.status(
            "[bold green]Retrieving checkpoint artifacts...", spinner="dots"
        ):
            target_path = train_cli.download_checkpoint_artifacts(
                remote_provider=remote_provider, job_id=job_id
            )
        console.print(
            f"✨ Training job checkpoint artifacts downloaded to {target_path}",
            style="bold green",
        )
    except Exception as e:
        error_console.print(f"Failed to download checkpoint artifacts data: {str(e)}")
        sys.exit(1)


@train.command(name="init")
@click.option("--list-examples", is_flag=True, help="List all available examples.")
@click.option("--target-directory", type=str, required=False)
@click.option("--examples", type=str, required=False)
@common.common_options()
def init_training_job(
    list_examples: Optional[bool],
    target_directory: Optional[str],
    examples: Optional[str],
) -> None:
    try:
        if list_examples:
            all_examples = train_cli._get_all_train_init_example_options()
            console.print("Available training examples:", style="bold")
            for example in all_examples:
                console.print(f"- {example}")
            console.print(
                "To launch, run `truss train init --examples <example1,example2>`",
                style="bold",
            )
            return

        selected_options = examples.split(",") if examples else []

        # No examples selected, initialize empty training project structure
        if not selected_options:
            if target_directory is None:
                target_directory = "truss-train-init"
            console.print(f"Initializing empty training project at {target_directory}")
            os.makedirs(target_directory)
            copy_tree_path(Path(TRAINING_TEMPLATE_DIR), Path(target_directory))
            console.print(
                f"✨ Empty training project initialized at {target_directory}",
                style="bold green",
            )
            return

        if target_directory is None:
            target_directory = os.getcwd()
        for example_to_download in selected_options:
            download_info = train_cli._get_train_init_example_info(
                example_name=example_to_download
            )
            local_dir = os.path.join(target_directory, example_to_download)

            if not download_info:
                all_examples = train_cli._get_all_train_init_example_options()
                error_console.print(
                    f"Example {example_to_download} not found in the ml-cookbook repository. Examples have to be one or more comma separated values from: {', '.join(all_examples)}"
                )
                continue
            success = train_cli.download_git_directory(
                git_api_url=download_info[0]["url"], local_dir=local_dir
            )
            if success:
                console.print(
                    f"✨ Training directory for {example_to_download} initialized at {local_dir}",
                    style="bold green",
                )
            else:
                error_console.print(
                    f"Failed to initialize training artifacts to {local_dir}"
                )

    except Exception as e:
        error_console.print(f"Failed to initialize training artifacts: {str(e)}")
        sys.exit(1)


@train.group(name="cache")
def cache():
    """Cache-related subcommands for truss train"""


@cache.command(name="summarize")
@click.argument("project", type=str, required=True)
@click.option("--remote", type=str, required=False, help="Remote to use")
@click.option(
    "--sort",
    type=click.Choice(
        [
            SORT_BY_FILEPATH,
            SORT_BY_SIZE,
            SORT_BY_MODIFIED,
            SORT_BY_TYPE,
            SORT_BY_PERMISSIONS,
        ]
    ),
    default=SORT_BY_FILEPATH,
    help="Sort files by filepath, size, modified date, file type, or permissions.",
)
@click.option(
    "--order",
    type=click.Choice([SORT_ORDER_ASC, SORT_ORDER_DESC]),
    default=SORT_ORDER_ASC,
    help="Sort order: ascending or descending.",
)
@click.option(
    "-o",
    "--output-format",
    type=click.Choice([OUTPUT_FORMAT_CLI_TABLE, OUTPUT_FORMAT_CSV, OUTPUT_FORMAT_JSON]),
    default=OUTPUT_FORMAT_CLI_TABLE,
    help="Output format: cli-table (default), csv, or json.",
)
@common.common_options()
def view_cache_summary(
    project: str, remote: Optional[str], sort: str, order: str, output_format: str
):
    """View cache summary for a training project"""
    if not remote:
        remote = remote_cli.inquire_remote_name()

    remote_provider: BasetenRemote = cast(
        BasetenRemote, RemoteFactory.create(remote=remote)
    )

    train_cli.view_cache_summary_by_project(
        remote_provider, project, sort, order, output_format
    )


def _maybe_resolve_project_id_from_id_or_name(
    remote_provider: BasetenRemote, project_id: Optional[str], project: Optional[str]
) -> Optional[str]:
    """resolve the project_id or project. `project` can be name or id"""
    if project and project_id:
        console.print("Both `project-id` and `project` provided. Using `project`.")
    project_str = project or project_id
    if not project_str:
        return None
    return train_cli.fetch_project_by_name_or_id(remote_provider, project_str)["id"]


def _get_or_create_default_local_whetstone_project_id(remote_provider: BasetenRemote) -> str:
    projects = remote_provider.api.list_training_projects()
    for existing_project in projects:
        if existing_project.get("name") == DEFAULT_LOCAL_WHETSTONE_PROJECT_NAME:
            return existing_project["id"]

    # Bootstrap a default project so plain `truss train sft` can create jobs without
    # requiring explicit --project/--project-id arguments.
    created_project = remote_provider.upsert_training_project(
        TrainingProject(
            name=DEFAULT_LOCAL_WHETSTONE_PROJECT_NAME,
            job=TrainingJob(image=Image(base_image="busybox:1.36")),
        )
    )
    console.print(
        f"Created default local project '{DEFAULT_LOCAL_WHETSTONE_PROJECT_NAME}'.",
        style="green",
    )
    return created_project["id"]
