from __future__ import annotations

import json
import shutil  # noqa: F401 - compatibility re-export for existing integrations
import subprocess  # noqa: F401 - compatibility re-export for existing integrations
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Set, Tuple

import rich_click as click

from truss.cli.cannery.binary import resolve_cannery_binary  # noqa: F401
from truss.cli.cannery.errors import CanneryProtocolError  # noqa: F401
from truss.cli.cannery.runner import run_cannery as _run_cannery
from truss.cli.cli import truss_cli
from truss.cli.utils import common
from truss.cli.utils.output import console, console_to_stderr

_MAX_METADATA_PAGES = 10_000


def run_cannery(arguments, remote: Optional[str] = None):
    return _run_cannery(arguments, remote=remote)


def _output_option(function):
    return click.option(
        "--output",
        "output_format",
        type=click.Choice(["text", "json"]),
        default="text",
        show_default=True,
        help="Output format. Progress and status always go to stderr.",
    )(function)


def _remote_option(function):
    return click.option("--remote", type=str, required=False, help="Remote to use.")(
        function
    )


def _page_size_option(function):
    return click.option(
        "--page-size",
        type=click.IntRange(1, 1_000),
        required=False,
        help="Metadata page size. Truss follows all returned pages.",
    )(function)


def _clean_json_stdout(function: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("output_format") == "json":
            with console_to_stderr():
                return function(*args, **kwargs)
        return function(*args, **kwargs)

    return wrapper


def _emit_result(result: Mapping[str, Any], output_format: str) -> None:
    if output_format == "json":
        click.echo(json.dumps(result))
    else:
        console.print_json(data=dict(result))


def _run_all_metadata_pages(
    arguments: list[str], remote: Optional[str], *, command: str
) -> Dict[str, Any]:
    result: Optional[Dict[str, Any]] = None
    next_page_token: Optional[str] = None
    seen_tokens: Set[str] = set()
    for _ in range(_MAX_METADATA_PAGES):
        page_arguments = list(arguments)
        if next_page_token is not None:
            page_arguments.extend(["--page-token", next_page_token])
        page_result = run_cannery(page_arguments, remote=remote)
        if result is None:
            result = page_result
        else:
            _append_metadata_page(result, page_result, command)

        page = _metadata_page(page_result, command)
        raw_token = page.get("next_page_token")
        if raw_token is None:
            _metadata_page(result, command).pop("next_page_token", None)
            return result
        if not isinstance(raw_token, str) or not raw_token:
            raise CanneryProtocolError(
                "Cannery metadata returned an invalid next_page_token."
            )
        if raw_token in seen_tokens:
            raise CanneryProtocolError(
                "Cannery metadata returned a repeated next_page_token."
            )
        seen_tokens.add(raw_token)
        next_page_token = raw_token
    raise CanneryProtocolError("Cannery metadata exceeded the pagination limit.")


def _metadata_page(result: Mapping[str, Any], command: str) -> Dict[str, Any]:
    if command == "ls":
        if "namespaces" in result or "references" in result:
            return result  # type: ignore[return-value]
    elif command == "show":
        file_page = result.get("file_page")
        if isinstance(file_page, dict):
            return file_page
    raise CanneryProtocolError(
        f"Cannery {command} result is missing its metadata page."
    )


def _append_metadata_page(
    result: Dict[str, Any], page_result: Dict[str, Any], command: str
) -> None:
    destination_page = _metadata_page(result, command)
    source_page = _metadata_page(page_result, command)
    entries_key = (
        "files"
        if command == "show"
        else "namespaces"
        if "namespaces" in destination_page
        else "references"
    )
    if entries_key not in source_page:
        raise CanneryProtocolError(
            "Cannery metadata page type changed during pagination."
        )
    destination_entries = destination_page.get(entries_key)
    source_entries = source_page.get(entries_key)
    if not isinstance(destination_entries, list) or not isinstance(
        source_entries, list
    ):
        raise CanneryProtocolError("Cannery metadata page entries must be arrays.")
    destination_entries.extend(source_entries)

    ignored_keys = {"correlation_id"}
    if command == "show":
        ignored_keys.add("file_page")
    else:
        ignored_keys.update({"namespaces", "references", "next_page_token"})
    for key, value in result.items():
        if key not in ignored_keys and page_result.get(key) != value:
            raise CanneryProtocolError(
                "Cannery metadata identity changed during pagination."
            )


@click.group()
def volume() -> None:
    """Manage Cannery-backed volumes."""


truss_cli.add_command(volume)


@volume.command(name="push")
@click.argument("path", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("ref", required=False)
@_output_option
@_remote_option
@_clean_json_stdout
@common.common_options()
def push_volume(
    path: Path, ref: Optional[str], output_format: str, remote: Optional[str]
) -> None:
    """Upload PATH as a volume, optionally naming it with REF."""
    arguments = ["push", str(path)]
    if ref is not None:
        arguments.append(ref)
    _emit_result(run_cannery(arguments, remote=remote), output_format)


@volume.command(name="ls")
@click.argument("namespace", required=False)
@click.option("--all", "include_all", is_flag=True, help="Include untagged versions.")
@_page_size_option
@_output_option
@_remote_option
@_clean_json_stdout
@common.common_options()
def list_volumes(
    namespace: Optional[str],
    include_all: bool,
    page_size: Optional[int],
    output_format: str,
    remote: Optional[str],
) -> None:
    """List namespaces, or volumes in NAMESPACE."""
    arguments = ["ls"]
    if namespace is not None:
        arguments.append(namespace)
    if include_all:
        arguments.append("--all")
    if page_size is not None:
        arguments.extend(["--page-size", str(page_size)])
    _emit_result(
        _run_all_metadata_pages(arguments, remote, command="ls"), output_format
    )


@volume.command(name="show")
@click.argument("ref")
@_page_size_option
@_output_option
@_remote_option
@_clean_json_stdout
@common.common_options()
def show_volume(
    ref: str, page_size: Optional[int], output_format: str, remote: Optional[str]
) -> None:
    """Show metadata and files for REF."""
    arguments = ["show", ref]
    if page_size is not None:
        arguments.extend(["--page-size", str(page_size)])
    _emit_result(
        _run_all_metadata_pages(arguments, remote, command="show"), output_format
    )


@volume.command(name="pull")
@click.argument("ref")
@click.argument("out_dir", type=click.Path(file_okay=False, path_type=Path))
@click.option(
    "--include",
    "include_paths",
    multiple=True,
    help="Volume-relative file or directory to pull. May be repeated.",
)
@_output_option
@_remote_option
@_clean_json_stdout
@common.common_options()
def pull_volume(
    ref: str,
    out_dir: Path,
    include_paths: Tuple[str, ...],
    output_format: str,
    remote: Optional[str],
) -> None:
    """Download files from REF into OUT_DIR."""
    if any(not include_path for include_path in include_paths):
        raise click.BadParameter("must not be empty", param_hint="'--include'")
    arguments = ["pull", ref, str(out_dir)]
    for include_path in include_paths:
        arguments.extend(["--include", include_path])
    _emit_result(run_cannery(arguments, remote=remote), output_format)
