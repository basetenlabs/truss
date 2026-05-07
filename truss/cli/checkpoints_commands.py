"""``truss checkpoints`` command group.

Surfaces the checkpoint-flavored pieces of ``truss train`` under Loops
language. Each command is a thin wrapper around a shared implementation
in ``train_commands`` so behavior stays in lockstep until the legacy
``truss train`` aliases are removed in a follow-up release.
"""

from typing import Optional

import rich_click as click

from truss.cli.cli import truss_cli
from truss.cli.train import checkpoint_viewer as checkpoint_mod
from truss.cli.train.cache import SORT_ORDER_ASC, SORT_ORDER_DESC
from truss.cli.train_commands import _run_deploy_checkpoints, _run_view_checkpoints
from truss.cli.utils import common


@click.group()
def checkpoints():
    """Subcommands for working with Loops and training-job checkpoints."""


truss_cli.add_command(checkpoints)


@checkpoints.command(name="deploy")
@click.option("--project-id", type=str, required=False, help="Project ID.")
@click.option("--project", type=str, required=False, help="Project name or project id.")
@click.option("--job-id", type=str, required=False, help="Job ID.")
@click.option(
    "--run-id",
    type=str,
    required=False,
    help=(
        "Loops run ID. Use to deploy checkpoints from a Loops run instead of "
        "a training job."
    ),
)
@click.option(
    "--config",
    type=str,
    required=False,
    help="Path to a Python file that defines a DeployCheckpointsConfig.",
)
@click.option(
    "--dry-run", is_flag=True, help="Generate a truss config without deploying."
)
@click.option(
    "--truss-config-output-dir",
    type=str,
    required=False,
    help=(
        "Path to output the truss config to. If not provided, will output to "
        "truss_configs/<model_version_name>_<model_version_id> or "
        "truss_configs/dry_run_<timestamp> if dry run."
    ),
)
@click.option("--remote", type=str, required=False, help="Remote to use.")
@common.common_options()
def deploy_checkpoints_cmd(
    project_id: Optional[str],
    project: Optional[str],
    job_id: Optional[str],
    run_id: Optional[str],
    config: Optional[str],
    remote: Optional[str],
    dry_run: bool,
    truss_config_output_dir: Optional[str],
) -> None:
    """Deploy a checkpoint via vLLM.

    Routes to the Loops backend when ``--run-id`` (or
    ``checkpoint_details.trainer_checkpoint_ids`` in ``--config``) is set,
    and to the training-jobs backend otherwise.
    """
    _run_deploy_checkpoints(
        project_id=project_id,
        project=project,
        job_id=job_id,
        run_id=run_id,
        config=config,
        remote=remote,
        dry_run=dry_run,
        truss_config_output_dir=truss_config_output_dir,
    )


@checkpoints.command(name="view")
@click.option("--remote", type=str, required=False, help="Remote to use.")
@click.option("--project-id", type=str, required=False, help="Project ID.")
@click.option("--project", type=str, required=False, help="Project name or project id.")
@click.option("--job-id", type=str, required=False, help="Job ID.")
@click.option(
    "--run-id",
    type=str,
    required=False,
    help="Loops run ID. View checkpoints saved by this Loops run.",
)
@click.option(
    "--model-name",
    type=str,
    required=False,
    help=(
        "Base model name. View Loops checkpoints associated with this base "
        "model across the caller's runs."
    ),
)
@click.option(
    "--checkpoint-name",
    type=str,
    required=False,
    help="Jump directly into a specific checkpoint's files.",
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
    type=click.Choice([SORT_ORDER_ASC, SORT_ORDER_DESC]),
    default=SORT_ORDER_ASC,
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
@common.common_options()
def view_checkpoints_cmd(
    remote: Optional[str],
    project_id: Optional[str],
    project: Optional[str],
    job_id: Optional[str],
    run_id: Optional[str],
    model_name: Optional[str],
    checkpoint_name: Optional[str],
    sort: str,
    order: str,
    output_format: str,
) -> None:
    """View checkpoints for a Loops run, base model, or training job.

    Routes to the Loops backend when ``--run-id`` or ``--model-name`` is
    set, and to the training-jobs backend otherwise.
    """
    _run_view_checkpoints(
        remote=remote,
        project_id=project_id,
        project=project,
        job_id=job_id,
        run_id=run_id,
        model_name=model_name,
        checkpoint_name=checkpoint_name,
        sort=sort,
        order=order,
        output_format=output_format,
    )
