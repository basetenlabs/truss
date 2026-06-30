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
    ScopeKind,
    _get_sort_key,
)
from truss.cli.utils.output import console
from truss.remote.baseten.remote import BasetenRemote


def view_loops_checkpoint_list(
    remote_provider: BasetenRemote,
    run_id: Optional[str] = None,
    base_model: Optional[str] = None,
    sort_by: str = SORT_BY_CREATED,
    order: str = SORT_ORDER_ASC,
    output_format: str = OUTPUT_FORMAT_CLI_TABLE,
) -> None:
    """View Loops checkpoints for a single run or across all runs of a base model.

    Mirrors `view_checkpoint_list` for training jobs but hits the Loops
    checkpoint endpoint, which returns each checkpoint's own run id so a
    base-model query can span multiple runs. Interactive file drill-down is
    not yet supported because Loops files are listed per-checkpoint rather
    than per-run.
    """
    scope_kind = ScopeKind.RUN if run_id is not None else ScopeKind.BASE_MODEL
    scope_id = run_id if run_id is not None else base_model
    assert scope_id is not None  # caller validates exactly one is set

    viewer: CheckpointListViewer
    if output_format == OUTPUT_FORMAT_CSV:
        viewer = CSVCheckpointViewer(scope_kind=scope_kind)
    elif output_format == OUTPUT_FORMAT_JSON:
        viewer = JSONCheckpointViewer(scope_kind=scope_kind)
    elif output_format == OUTPUT_FORMAT_CLI_TABLE:
        viewer = CLITableCheckpointViewer(scope_kind=scope_kind)
    else:
        raise ValueError(f"Invalid output format: {output_format}")

    try:
        raw = remote_provider.api.list_loops_checkpoints(
            run_id=run_id, base_model=base_model
        )
        checkpoints = raw.get("checkpoints", [])

        reverse = order == SORT_ORDER_DESC
        sort_key = _get_sort_key(sort_by)
        checkpoints.sort(key=sort_key, reverse=reverse)

        if not checkpoints:
            viewer.output_no_checkpoints_message(scope_id)
        else:
            viewer.output_checkpoints(checkpoints, scope_id)
    except Exception as e:
        # Keep stdout clean for json/csv consumers (jq pipelines, spreadsheets);
        # only the CLI table format gets a styled error.
        if output_format in (OUTPUT_FORMAT_CSV, OUTPUT_FORMAT_JSON):
            print(f"Error fetching checkpoints: {e}", file=sys.stderr)
        else:
            console.print(f"Error fetching checkpoints: {e}", style="red")
        raise
