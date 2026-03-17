import csv
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Optional

import requests
import rich
from InquirerPy.prompts.fuzzy import FuzzyPrompt, InquirerPyFuzzyControl
from InquirerPy.utils import get_style
from prompt_toolkit import Application
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import Window
from prompt_toolkit.layout.controls import FormattedTextControl
from pygments import highlight
from pygments.formatters import TerminalFormatter
from pygments.lexers import get_lexer_for_filename

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

# Safetensor headers contain JSON metadata on layer names, shapes, and dtypes.
MAX_SAFETENSOR_HEADER_SIZE = 10_000_000

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


def _build_directory_listing(
    files: list[dict],
    current_path: str,
    checkpoint_lookup: Optional[dict[str, dict]] = None,
) -> tuple[list[dict], list[dict]]:
    """Build a listing of immediate child directories and files for current_path.

    Args:
        files: List of file dicts, each with a "_rel_path" key.
        current_path: Current directory path (e.g. "" for root, "rank-0" for inside rank-0).
        checkpoint_lookup: Optional mapping from checkpoint_id to metadata
            (checkpoint_type, base_model, size_bytes). Directories matching a
            checkpoint_id will be annotated with these fields.

    Returns:
        Tuple of (dirs, dir_files):
        - dirs: list of {"name": str, "total_size": int, "file_count": int,
          and optionally "checkpoint_type", "base_model", "size_bytes"}
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

    # Annotate directories that match detected checkpoints
    if checkpoint_lookup:
        for d in dir_stats.values():
            ckpt_meta = checkpoint_lookup.get(d["name"])
            if ckpt_meta:
                d["checkpoint_type"] = ckpt_meta.get("checkpoint_type", "")
                d["base_model"] = ckpt_meta.get("base_model", "")
                d["size_bytes"] = ckpt_meta.get("size_bytes", 0)

    return list(dir_stats.values()), dir_files


_DIR_STYLE = get_style({"dir": "#30B77E", "fuzzy_match": "#c678dd bold"})


class _ColoredFuzzyControl(InquirerPyFuzzyControl):
    """Fuzzy control that colors directory choices."""

    def _get_style_for_choice(self, choice) -> str:
        value = choice.get("value")
        if isinstance(value, tuple) and len(value) >= 1 and value[0] == "dir":
            return "class:dir"
        return ""

    def _get_normal_text(self, choice):
        display_choices = []
        display_choices.append(("class:pointer", len(self._pointer) * " "))
        display_choices.append(
            (
                "class:marker",
                self._marker
                if self.choices[choice["index"]]["enabled"]
                else self._marker_pl,
            )
        )
        style = self._get_style_for_choice(choice)
        if not choice["indices"]:
            display_choices.append((style, choice["name"]))
        else:
            indices = set(choice["indices"])
            for index, char in enumerate(choice["name"]):
                if index in indices:
                    display_choices.append(("class:fuzzy_match", char))
                else:
                    display_choices.append((style, char))
        return display_choices

    def _get_hover_text(self, choice):
        display_choices = []
        display_choices.append(("class:pointer", self._pointer))
        display_choices.append(
            (
                "class:marker",
                self._marker
                if self.choices[choice["index"]]["enabled"]
                else self._marker_pl,
            )
        )
        display_choices.append(("[SetCursorPosition]", ""))
        style = self._get_style_for_choice(choice) or "class:pointer"
        if not choice["indices"]:
            display_choices.append((style, choice["name"]))
        else:
            indices = set(choice["indices"])
            for index, char in enumerate(choice["name"]):
                if index in indices:
                    display_choices.append(("class:fuzzy_match", char))
                else:
                    display_choices.append((style, char))
        return display_choices


def _colored_fuzzy(*, choices: list[dict], allow_back: bool = True, **kwargs) -> Any:
    """Create a fuzzy prompt that colors directory choices.

    Right-arrow on a file shows its download URL.
    """
    if allow_back:
        instruction = "← back  → open/url  ↑↓ navigate  enter select  ctrl-c quit"
    else:
        instruction = "→ enter  ↑↓ navigate  enter select  ctrl-c quit"

    prompt = FuzzyPrompt(
        choices=choices, style=_DIR_STYLE, long_instruction=instruction, **kwargs
    )
    prompt._content_control.__class__ = _ColoredFuzzyControl

    @prompt.register_kb("right")
    def _right_action(event):
        control = prompt._content_control
        idx = control._selected_choice_index
        filtered = control._filtered_choices
        if 0 <= idx < len(filtered):
            value = filtered[idx].get("value")
            if isinstance(value, tuple) and value[0] == "file":
                event.app.exit(result=("url", value[1].get("url", "")))
            elif isinstance(value, tuple) and value[0] in ("dir", "checkpoint"):
                event.app.exit(result=value)

    if allow_back:

        @prompt.register_kb("left")
        def _go_back(event):
            event.app.exit(result=("back", None))

    return prompt


def _show_url(url: str) -> None:
    """Display a download URL and wait for left-arrow to dismiss."""
    kb = KeyBindings()

    @kb.add("left")
    @kb.add("escape")
    @kb.add("q")
    @kb.add("enter")
    def _exit(event):
        event.app.exit()

    text: list[tuple[str, str]] = [
        ("", "Download URL:\n\n"),
        ("underline", url),
        ("#888888", "\n\nPress left arrow to go back"),
    ]
    app: Application = Application(
        layout=Layout(Window(FormattedTextControl(text))),  # type: ignore[arg-type]
        key_bindings=kb,
        full_screen=True,
    )
    app.run()


def _build_explorer_choices(
    dirs: list[dict], dir_files: list[dict], has_parent: bool
) -> list[dict]:
    """Build the list of fuzzy-prompt choices for the file explorer."""
    choices: list[dict] = []
    if has_parent:
        choices.append({"name": "..", "value": ("back", None)})
    for d in sorted(dirs, key=lambda x: x["name"]):
        size_str = cli_common.format_bytes_to_human_readable(d["total_size"])
        label = f"{d['name']}/ ({size_str}, {d['file_count']} files)"
        if d.get("checkpoint_type"):
            ckpt_type = d["checkpoint_type"]
            base_model = d.get("base_model", "")
            annotation_parts = [ckpt_type]
            if base_model:
                annotation_parts.append(base_model)
            sep = " \u00b7 "
            label += f" [{sep.join(annotation_parts)}]"
        choices.append({"name": label, "value": ("dir", d["name"])})
    for f in sorted(dir_files, key=lambda x: x["_rel_path"]):
        name = f["_rel_path"].split("/")[-1]
        size_str = cli_common.format_bytes_to_human_readable(f.get("size_bytes", 0))
        choices.append({"name": f"{name} ({size_str})", "value": ("file", f)})
    choices.append({"name": EXIT_OPTION, "value": ("exit", None)})
    return choices


def _explore_files(
    files: list[dict],
    job_id: str,
    checkpoint_lookup: Optional[dict[str, dict]] = None,
    initial_path: Optional[str] = None,
) -> None:
    """Interactive file explorer starting from the root of the checkpoint volume."""
    prepared_files = [
        {**f, "_rel_path": f.get("relative_file_name", "")} for f in files
    ]

    path_stack: list[str] = (
        [p for p in initial_path.split("/") if p and p != "."] if initial_path else []
    )
    min_depth = len(path_stack)

    while True:
        current_path = "/".join(path_stack)
        dirs, dir_files = _build_directory_listing(
            prepared_files, current_path, checkpoint_lookup
        )

        console.clear()
        display_path = job_id + "/" + current_path if current_path else job_id + "/"
        console.print(display_path, style="bold cyan")

        choices = _build_explorer_choices(
            dirs, dir_files, has_parent=len(path_stack) > min_depth
        )

        selected = _colored_fuzzy(
            message="", qmark="", prompt="Navigate: ", choices=choices
        ).execute()
        action, payload = selected

        if action == "dir":
            path_stack.append(payload)
        elif action == "file":
            file_name = payload.get("relative_file_name", "")
            if file_name.endswith(".safetensors"):
                _view_safetensor_file(payload)
            elif payload.get("size_bytes", 0) >= MAX_VIEWABLE_FILE_SIZE:
                _show_url(payload.get("url", ""))
            else:
                _fetch_and_display_file(payload)
        elif action == "url":
            _show_url(payload)
        elif action == "back":
            if len(path_stack) > min_depth:
                path_stack.pop()
            else:
                return
        elif action == "exit":
            return


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


def _highlight_content(content: str, file_name: str) -> str:
    """Apply syntax highlighting with ANSI escape codes if a lexer is available."""
    try:
        lexer = get_lexer_for_filename(file_name)
        return highlight(content, lexer, TerminalFormatter())
    except Exception:
        return content


def _open_in_pager(content: str, file_name: str) -> None:
    """Open content in the user's pager. Falls back to printing if no pager is available."""
    pager_cmd = _get_pager_command()
    if not pager_cmd:
        console.print(content)
        return

    highlighted = _highlight_content(content, file_name)

    suffix = os.path.splitext(file_name)[1] or ".txt"
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=suffix, prefix="ckpt-", delete=False
    ) as tmp:
        tmp.write(highlighted)
        tmp_path = tmp.name

    try:
        subprocess.run([*pager_cmd, tmp_path])
    finally:
        os.unlink(tmp_path)


DTYPE_SIZES = {
    "F64": 8,
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "I64": 8,
    "I32": 4,
    "I16": 2,
    "I8": 1,
    "U8": 1,
    "BOOL": 1,
}


@dataclass
class TensorSummary:
    name: str
    dtype: str
    shape: list[int]
    num_params: int
    size_bytes: Optional[int]

    @staticmethod
    def from_header_entry(name: str, info: dict) -> "TensorSummary":
        dtype = info.get("dtype", "?")
        shape = info.get("shape", [])
        num_params = 1
        for dim in shape:
            num_params *= dim
        dtype_size = DTYPE_SIZES.get(dtype)
        size_bytes = num_params * dtype_size if dtype_size is not None else None
        return TensorSummary(name, dtype, shape, num_params, size_bytes)


@dataclass
class SafetensorSummary:
    metadata: dict[str, str]
    tensors: list[TensorSummary]

    @staticmethod
    def from_header(header: dict) -> "SafetensorSummary":
        metadata = header.pop("__metadata__", {})
        tensors = [
            TensorSummary.from_header_entry(name, info)
            for name, info in sorted(header.items())
        ]
        return SafetensorSummary(metadata, tensors)

    def __str__(self) -> str:
        total_params = sum(t.num_params for t in self.tensors)
        total_bytes = sum(
            t.size_bytes for t in self.tensors if t.size_bytes is not None
        )

        lines: list[str] = []
        if self.metadata:
            lines.append("Metadata:")
            for k, v in sorted(self.metadata.items()):
                lines.append(f"  {k}: {v}")
            lines.append("")

        lines.append(
            f"Tensors: {len(self.tensors)}  |  Parameters: {total_params:,}  |  Size: {cli_common.format_bytes_to_human_readable(total_bytes)}"
        )
        lines.append("")

        name_width = max((len(t.name) for t in self.tensors), default=4)
        name_width = max(name_width, 4)
        lines.append(
            f"{'Name':<{name_width}}  {'Dtype':<6}  {'Shape':<20}  {'Params':>12}  {'Size':>10}"
        )
        lines.append("-" * (name_width + 56))

        for t in self.tensors:
            shape_str = str(t.shape)
            size_str = (
                cli_common.format_bytes_to_human_readable(t.size_bytes)
                if t.size_bytes is not None
                else "?"
            )
            lines.append(
                f"{t.name:<{name_width}}  {t.dtype:<6}  {shape_str:<20}  {t.num_params:>12,}  {size_str:>10}"
            )

        return "\n".join(lines)


def _fetch_safetensor_header(url: str) -> Optional[dict]:
    """Fetch only the header of a safetensors file using range requests."""
    try:
        size_resp = requests.get(url, headers={"Range": "bytes=0-7"}, timeout=10)
        size_resp.raise_for_status()
        header_size = struct.unpack("<Q", size_resp.content)[0]

        if header_size > MAX_SAFETENSOR_HEADER_SIZE:
            return None

        header_resp = requests.get(
            url, headers={"Range": f"bytes=8-{8 + header_size - 1}"}, timeout=30
        )
        header_resp.raise_for_status()
        return json.loads(header_resp.content)
    except Exception:
        return None


def _view_safetensor_file(file_info: dict) -> None:
    """Fetch and display safetensor layer information."""
    url = file_info.get("url", "")
    file_name = file_info.get("relative_file_name", "")

    with console.status("Fetching safetensor header...", spinner="dots"):
        header = _fetch_safetensor_header(url)

    if header is None:
        console.print("Failed to read safetensor header.", style="red")
        input("Press Enter to continue...")
        return

    summary = SafetensorSummary.from_header(header)
    content = f"# {file_name}\n\n" + str(summary)
    _open_in_pager(content, file_name)


def _fetch_and_display_file(file_info: dict) -> None:
    """Fetch a file via its presigned URL and open it in a pager."""
    url = file_info.get("url", "")
    file_name = file_info.get("relative_file_name", "")

    if file_name.endswith(".safetensors"):
        _view_safetensor_file(file_info)
        return

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
        input("Press Enter to continue...")


def _select_checkpoint(checkpoints: list[dict], job_id: str) -> Optional[str]:
    """Prompt the user to select a checkpoint. Returns checkpoint_id or None to exit."""
    choices = []
    for ckpt in checkpoints:
        cid = ckpt.get("checkpoint_id", "")
        ckpt_type = ckpt.get("checkpoint_type", "")
        size_str = cli_common.format_bytes_to_human_readable(ckpt.get("size_bytes", 0))
        created_str = cli_common.format_localized_time(ckpt.get("created_at", ""))
        label = f"{cid}  [{ckpt_type}]  {size_str}  {created_str}"
        choices.append({"name": label, "value": ("checkpoint", cid)})
    choices.append({"name": EXIT_OPTION, "value": ("exit", None)})

    console.clear()
    console.print(f"Checkpoints for job: {job_id}", style="bold cyan")
    result = _colored_fuzzy(
        message="",
        qmark="",
        prompt="Select checkpoint: ",
        choices=choices,
        allow_back=False,
    ).execute()
    action, payload = result
    if action == "checkpoint":
        return payload
    return None


def view_checkpoint_list(
    remote_provider: BasetenRemote,
    project_id: str,
    job_id: str,
    sort_by: str = SORT_BY_CREATED,
    order: str = SORT_ORDER_ASC,
    output_format: str = OUTPUT_FORMAT_CLI_TABLE,
    interactive: bool = False,
    checkpoint_name: Optional[str] = None,
) -> None:
    """View checkpoints for a training job, with optional interactive drill-down."""
    viewer_factories: dict[str, type[CheckpointListViewer]] = {
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

        reverse = order == SORT_ORDER_DESC
        sort_key = _get_sort_key(sort_by)
        checkpoints.sort(key=sort_key, reverse=reverse)

        # Build checkpoint lookup for annotations
        checkpoint_lookup: dict[str, dict] = {}
        for ckpt in checkpoints:
            cid = ckpt.get("checkpoint_id", "")
            if cid and cid != ".":
                checkpoint_lookup[cid] = {
                    "checkpoint_type": ckpt.get("checkpoint_type", ""),
                    "base_model": ckpt.get("base_model", ""),
                    "size_bytes": ckpt.get("size_bytes", 0),
                }

        if checkpoint_name:
            with console.status("Fetching checkpoint files...", spinner="dots"):
                all_files = (
                    remote_provider.api.get_training_job_checkpoint_presigned_url(
                        project_id, job_id, page_size=1000
                    )
                )
            if not all_files:
                console.print("No files found in checkpoint volume.", style="yellow")
                return
            if not any(
                f.get("relative_file_name", "").startswith(f"{checkpoint_name}/")
                for f in all_files
            ):
                console.print(
                    f"No files found for checkpoint: {checkpoint_name}", style="yellow"
                )
                return
            _explore_files(
                all_files,
                job_id,
                checkpoint_lookup or None,
                initial_path=checkpoint_name,
            )
            return

        if not interactive or output_format != OUTPUT_FORMAT_CLI_TABLE:
            if not checkpoints:
                viewer.output_no_checkpoints_message(job_id)
            else:
                viewer.output_checkpoints(checkpoints, job_id)
            return

        if not checkpoints:
            viewer.output_no_checkpoints_message(job_id)
            return

        with console.status("Fetching checkpoint files...", spinner="dots"):
            all_files = remote_provider.api.get_training_job_checkpoint_presigned_url(
                project_id, job_id, page_size=1000
            )

        if not all_files:
            console.print("No files found in checkpoint volume.", style="yellow")
            return

        while True:
            selected_checkpoint = _select_checkpoint(checkpoints, job_id)
            if selected_checkpoint is None:
                return
            _explore_files(
                all_files,
                job_id,
                checkpoint_lookup or None,
                initial_path=selected_checkpoint,
            )

    except Exception as e:
        if output_format in (OUTPUT_FORMAT_CSV, OUTPUT_FORMAT_JSON):
            print(f"Error fetching checkpoints: {str(e)}", file=sys.stderr)
        else:
            console.print(f"Error fetching checkpoints: {str(e)}", style="red")
        raise
