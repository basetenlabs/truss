import sys
from typing import Optional

from truss.cli.train.checkpoint_viewer import (
    OUTPUT_FORMAT_CLI_TABLE,
    OUTPUT_FORMAT_CSV,
    OUTPUT_FORMAT_JSON,
    SORT_BY_CREATED,
    SORT_ORDER_ASC,
    SORT_ORDER_DESC,
    CheckpointListViewer,
    CLITableCheckpointViewer,
    CSVCheckpointViewer,
    JSONCheckpointViewer,
    _get_sort_key,
)
from truss.cli.utils.output import console
from truss.remote.baseten.remote import BasetenRemote


def view_loops_checkpoint_list(
    remote_provider: BasetenRemote,
    run_id: str,
    sort_by: str = SORT_BY_CREATED,
    order: str = SORT_ORDER_ASC,
    output_format: str = OUTPUT_FORMAT_CLI_TABLE,
) -> None:
    """View checkpoints for a Loops run.

    Mirrors `view_checkpoint_list` for training jobs but hits the Loops
    checkpoint endpoint. Interactive file drill-down is not yet supported
    because Loops files are listed per-checkpoint rather than per-run.
    """
    viewer: CheckpointListViewer
    if output_format == OUTPUT_FORMAT_CSV:
        viewer = CSVCheckpointViewer(scope_kind="run")
    elif output_format == OUTPUT_FORMAT_JSON:
        viewer = JSONCheckpointViewer(scope_kind="run")
    elif output_format == OUTPUT_FORMAT_CLI_TABLE:
        viewer = CLITableCheckpointViewer(scope_kind="run")
    else:
        raise ValueError(f"Invalid output format: {output_format}")

    try:
        raw = remote_provider.api.list_loops_checkpoints(run_id=run_id)
        checkpoints = raw.get("checkpoints", [])

        reverse = order == SORT_ORDER_DESC
        sort_key = _get_sort_key(sort_by)
        checkpoints.sort(key=sort_key, reverse=reverse)

        if not checkpoints:
            viewer.output_no_checkpoints_message(run_id)
        else:
            viewer.output_checkpoints(checkpoints, run_id)
    except Exception as e:
        # Keep stdout clean for json/csv consumers (jq pipelines, spreadsheets);
        # only the CLI table format gets a styled error.
        if output_format in (OUTPUT_FORMAT_CSV, OUTPUT_FORMAT_JSON):
            print(f"Error fetching checkpoints: {e}", file=sys.stderr)
        else:
            console.print(f"Error fetching checkpoints: {e}", style="red")
        raise


def resolve_run_id(
    remote_provider: BasetenRemote, run_id: Optional[str], base_model: Optional[str]
) -> str:
    """Resolve a Loops run from --run-id or --base-model.

    With --run-id, returns it as-is (existence is validated server-side
    on the next call). With --base-model, picks the most recently created
    run for that base model.
    """
    if run_id and base_model:
        raise ValueError("Pass either --run-id or --base-model, not both.")
    if run_id:
        return run_id
    if not base_model:
        raise ValueError("Pass --run-id or --base-model to identify a Loops run.")

    runs = remote_provider.api.list_loops_runs(base_model=base_model)
    if not runs:
        raise ValueError(f"No Loops runs found for base model: {base_model}")
    runs = sorted(runs, key=lambda r: r.get("created_at") or "", reverse=True)
    return runs[0]["id"]
