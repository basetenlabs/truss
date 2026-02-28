import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

import requests
import rich
from InquirerPy import inquirer

from truss.cli.utils import common as cli_common
from truss.cli.utils.output import console
from truss.remote.baseten.remote import BasetenRemote

# Sort constants
SORT_BY_CHECKPOINT_ID = "checkpoint-id"
SORT_BY_CREATED = "created"
SORT_BY_SIZE = "size"
SORT_BY_TYPE = "type"
SORT_ORDER_ASC = "asc"
SORT_ORDER_DESC = "desc"

# Output format constants
OUTPUT_FORMAT_CLI_TABLE = "cli-table"
OUTPUT_FORMAT_CSV = "csv"
OUTPUT_FORMAT_JSON = "json"

# Max file size for inline viewing (1 MB)
MAX_VIEWABLE_FILE_SIZE = 1_000_000

EXIT_OPTION = "[Exit]"


def _get_sort_key(sort_by: str) -> Callable[[dict], Any]:
    """Get the sort key function for the given sort option."""
    if sort_by == SORT_BY_CHECKPOINT_ID:
        return lambda x: x.get("checkpoint_id", "")
    elif sort_by == SORT_BY_CREATED:
        return lambda x: x.get("created_at", "")
    elif sort_by == SORT_BY_SIZE:
        return lambda x: x.get("size_bytes", 0)
    elif sort_by == SORT_BY_TYPE:
        return lambda x: x.get("checkpoint_type", "")
    else:
        raise ValueError(f"Invalid --sort argument: {sort_by}")


class CheckpointListViewer(ABC):
    """Base class for checkpoint list viewers that output in different formats."""

    @abstractmethod
    def output_checkpoints(self, checkpoints: list[dict], job_id: str) -> None:
        """Output the checkpoint list in the viewer's format."""
        pass

    @abstractmethod
    def output_no_checkpoints_message(self, job_id: str) -> None:
        """Output message when no checkpoints are found."""
        pass


class CLITableCheckpointViewer(CheckpointListViewer):
    """Viewer that outputs checkpoint list as a styled CLI table."""

    def output_checkpoints(self, checkpoints: list[dict], job_id: str) -> None:
        table = rich.table.Table(
            title=f"Checkpoints for job: {job_id}",
            show_header=True,
            header_style="bold magenta",
            box=rich.table.box.ROUNDED,
            border_style="blue",
        )
        table.add_column("Checkpoint ID", style="cyan")
        table.add_column("Type")
        table.add_column("Base Model", style="yellow")
        table.add_column("Size", style="green")
        table.add_column("Created At", style="dim")

        for ckpt in checkpoints:
            size_str = cli_common.format_bytes_to_human_readable(
                ckpt.get("size_bytes", 0)
            )
            created_str = cli_common.format_localized_time(ckpt.get("created_at", ""))
            table.add_row(
                ckpt.get("checkpoint_id", ""),
                ckpt.get("checkpoint_type", ""),
                ckpt.get("base_model", "") or "",
                size_str,
                created_str,
            )

        console.print(table)

    def output_no_checkpoints_message(self, job_id: str) -> None:
        console.print(f"No checkpoints found for job: {job_id}.", style="yellow")


class CSVCheckpointViewer(CheckpointListViewer):
    """Viewer that outputs checkpoint list in CSV format."""

    def output_checkpoints(self, checkpoints: list[dict], job_id: str) -> None:
        writer = csv.writer(sys.stdout)
        writer.writerow(
            [
                "Checkpoint ID",
                "Type",
                "Base Model",
                "Size (bytes)",
                "Size (human readable)",
                "Created At",
            ]
        )
        for ckpt in checkpoints:
            size_str = cli_common.format_bytes_to_human_readable(
                ckpt.get("size_bytes", 0)
            )
            created_str = cli_common.format_localized_time(ckpt.get("created_at", ""))
            writer.writerow(
                [
                    ckpt.get("checkpoint_id", ""),
                    ckpt.get("checkpoint_type", ""),
                    ckpt.get("base_model", "") or "",
                    str(ckpt.get("size_bytes", 0)),
                    size_str,
                    created_str,
                ]
            )

    def output_no_checkpoints_message(self, job_id: str) -> None:
        writer = csv.writer(sys.stdout)
        writer.writerow(
            [
                "Checkpoint ID",
                "Type",
                "Base Model",
                "Size (bytes)",
                "Size (human readable)",
                "Created At",
            ]
        )


class JSONCheckpointViewer(CheckpointListViewer):
    """Viewer that outputs checkpoint list in JSON format."""

    def output_checkpoints(self, checkpoints: list[dict], job_id: str) -> None:
        checkpoints_data = []
        for ckpt in checkpoints:
            size_str = cli_common.format_bytes_to_human_readable(
                ckpt.get("size_bytes", 0)
            )
            created_str = cli_common.format_localized_time(ckpt.get("created_at", ""))
            entry: dict[str, Any] = {
                "checkpoint_id": ckpt.get("checkpoint_id", ""),
                "checkpoint_type": ckpt.get("checkpoint_type", ""),
                "base_model": ckpt.get("base_model", "") or "",
                "size_bytes": ckpt.get("size_bytes", 0),
                "size_human_readable": size_str,
                "created_at": created_str,
            }
            if ckpt.get("lora_adapter_config"):
                entry["lora_adapter_config"] = ckpt["lora_adapter_config"]
            checkpoints_data.append(entry)

        output = {
            "job_id": job_id,
            "total_checkpoints": len(checkpoints),
            "checkpoints": checkpoints_data,
        }
        print(json.dumps(output, indent=2))

    def output_no_checkpoints_message(self, job_id: str) -> None:
        output = {"job_id": job_id, "total_checkpoints": 0, "checkpoints": []}
        print(json.dumps(output, indent=2))


def _select_checkpoint(checkpoints: list[dict]) -> Optional[str]:
    """Prompt user to select a checkpoint from the list."""
    choices = [ckpt["checkpoint_id"] for ckpt in checkpoints] + [EXIT_OPTION]
    selected = inquirer.select(
        message="Select a checkpoint to view files:", choices=choices
    ).execute()
    if selected == EXIT_OPTION:
        return None
    return selected


def _build_directory_listing(
    files: list[dict], current_path: str
) -> tuple[list[dict], list[dict]]:
    """Build a listing of immediate child directories and files for current_path.

    Args:
        files: List of file dicts, each with a "_rel_path" key (checkpoint prefix stripped).
        current_path: Current directory path (e.g. "" for root, "rank-0" for inside rank-0).

    Returns:
        Tuple of (dirs, dir_files):
        - dirs: list of {"name": str, "total_size": int, "file_count": int}
        - dir_files: list of original file dicts that are direct children of current_path
    """
    dir_stats: dict[str, dict] = {}
    dir_files: list[dict] = []

    prefix = current_path + "/" if current_path else ""

    for f in files:
        rel = f.get("_rel_path", "")
        if not rel.startswith(prefix):
            continue
        remainder = rel[len(prefix) :]
        if not remainder:
            continue

        if "/" in remainder:
            # File belongs to a subdirectory
            subdir_name = remainder.split("/", 1)[0]
            if subdir_name not in dir_stats:
                dir_stats[subdir_name] = {
                    "name": subdir_name,
                    "total_size": 0,
                    "file_count": 0,
                }
            dir_stats[subdir_name]["total_size"] += f.get("size_bytes", 0)
            dir_stats[subdir_name]["file_count"] += 1
        else:
            # Direct child file
            dir_files.append(f)

    return list(dir_stats.values()), dir_files


def _explore_checkpoint_files(files: list[dict], checkpoint_name: str) -> bool:
    """Interactive file explorer. Returns True to exit entirely, False to go back to checkpoint selection."""
    prefix = checkpoint_name + "/"
    stripped_files = []
    for f in files:
        rel = f.get("relative_file_name", "")
        if rel.startswith(prefix):
            rel = rel[len(prefix) :]
        stripped_files.append({**f, "_rel_path": rel})

    path_stack: list[str] = []

    while True:
        current_path = "/".join(path_stack)
        dirs, dir_files = _build_directory_listing(stripped_files, current_path)

        console.clear()
        display_path = (
            checkpoint_name + "/" + current_path
            if current_path
            else checkpoint_name + "/"
        )
        console.print(f"\U0001f4c2 {display_path}", style="bold cyan")

        # Build inquirer choices
        choices: list[dict] = [{"name": "..", "value": ("back", None)}]
        for d in sorted(dirs, key=lambda x: x["name"]):
            size_str = cli_common.format_bytes_to_human_readable(d["total_size"])
            choices.append(
                {
                    "name": f"\U0001f4c1 {d['name']}/ ({size_str}, {d['file_count']} files)",
                    "value": ("dir", d["name"]),
                }
            )
        for f in sorted(dir_files, key=lambda x: x["_rel_path"]):
            name = f["_rel_path"].split("/")[-1]
            size_str = cli_common.format_bytes_to_human_readable(f.get("size_bytes", 0))
            choices.append(
                {"name": f"\U0001f4c4 {name} ({size_str})", "value": ("file", f)}
            )
        choices.append({"name": EXIT_OPTION, "value": ("exit", None)})

        selected = inquirer.select(message="Navigate:", choices=choices).execute()
        action, payload = selected

        if action == "dir":
            path_stack.append(payload)
        elif action == "file":
            if payload.get("size_bytes", 0) >= MAX_VIEWABLE_FILE_SIZE:
                console.print(
                    "File too large to view inline (> 1 MB). Download URL:",
                    style="dim",
                )
                console.print(payload.get("url", ""), style="underline blue")
                console.input("[dim]Press Enter to continue...[/dim]")
            else:
                _fetch_and_display_file(payload)
        elif action == "back":
            if path_stack:
                path_stack.pop()
            else:
                return False
        elif action == "exit":
            return True


def _get_pager_command() -> list[str]:
    """Return a command list for a read-only pager/viewer."""
    pager_env = os.environ.get("PAGER")
    if pager_env:
        return pager_env.split()

    if shutil.which("less"):
        return ["less", "-R"]
    if shutil.which("more"):
        return ["more"]
    return []


def _open_in_pager(content: str, file_name: str) -> None:
    """Open content in the user's pager. Falls back to printing if no pager is available."""
    pager_cmd = _get_pager_command()
    if not pager_cmd:
        console.print(content)
        return

    suffix = os.path.splitext(file_name)[1] or ".txt"
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=suffix, prefix="ckpt-", delete=False
    ) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        subprocess.run([*pager_cmd, tmp_path])
    finally:
        os.unlink(tmp_path)


def _fetch_and_display_file(file_info: dict) -> None:
    """Fetch a file via its presigned URL and open it in a pager."""
    url = file_info.get("url", "")
    file_name = file_info.get("relative_file_name", "")

    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        content = resp.text

        if file_name.endswith(".json"):
            try:
                content = json.dumps(json.loads(content), indent=2)
            except json.JSONDecodeError:
                pass

        _open_in_pager(content, file_name)
    except requests.RequestException as e:
        console.print(f"Failed to fetch file: {e}", style="red")


def _filter_files_for_checkpoint(
    all_files: list[dict], checkpoint_name: str
) -> list[dict]:
    """Filter presigned URL files that belong to a specific checkpoint."""
    prefix = checkpoint_name + "/"
    return [f for f in all_files if f.get("relative_file_name", "").startswith(prefix)]


def view_checkpoint_list(
    remote_provider: BasetenRemote,
    project_id: str,
    job_id: str,
    sort_by: str = SORT_BY_CREATED,
    order: str = SORT_ORDER_ASC,
    output_format: str = OUTPUT_FORMAT_CLI_TABLE,
    interactive: bool = False,
) -> None:
    """View checkpoints for a training job, with optional interactive drill-down."""
    viewer_factories = {
        OUTPUT_FORMAT_CSV: CSVCheckpointViewer,
        OUTPUT_FORMAT_JSON: JSONCheckpointViewer,
        OUTPUT_FORMAT_CLI_TABLE: CLITableCheckpointViewer,
    }
    viewer_cls = viewer_factories.get(output_format)
    if not viewer_cls:
        raise ValueError(f"Invalid output format: {output_format}")
    viewer = viewer_cls()

    try:
        raw = remote_provider.api.list_training_job_checkpoints(project_id, job_id)
        checkpoints = raw.get("checkpoints", [])

        if not checkpoints:
            viewer.output_no_checkpoints_message(job_id)
            return

        reverse = order == SORT_ORDER_DESC
        sort_key = _get_sort_key(sort_by)
        checkpoints.sort(key=sort_key, reverse=reverse)

        viewer.output_checkpoints(checkpoints, job_id)

        if not interactive or output_format != OUTPUT_FORMAT_CLI_TABLE:
            return

        # Interactive drill-down loop
        all_files: Optional[list[dict]] = None
        while True:
            selected_id = _select_checkpoint(checkpoints)
            if selected_id is None:
                break

            # Lazy-fetch all checkpoint files on first drill-down
            if all_files is None:
                with console.status("Fetching checkpoint files...", spinner="dots"):
                    all_files = (
                        remote_provider.api.get_training_job_checkpoint_presigned_url(
                            project_id, job_id, page_size=1000
                        )
                    )

            files = _filter_files_for_checkpoint(all_files, selected_id)
            if not files:
                console.print(
                    f"No files found for checkpoint: {selected_id}", style="yellow"
                )
                continue

            should_exit = _explore_checkpoint_files(files, selected_id)
            if should_exit:
                return

    except Exception as e:
        if output_format in (OUTPUT_FORMAT_CSV, OUTPUT_FORMAT_JSON):
            print(f"Error fetching checkpoints: {str(e)}", file=sys.stderr)
        else:
            console.print(f"Error fetching checkpoints: {str(e)}", style="red")
        raise
