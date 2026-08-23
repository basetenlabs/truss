from __future__ import annotations

import json
import shutil  # noqa: F401 - compatibility re-export for existing integrations
import subprocess  # noqa: F401 - compatibility re-export for existing integrations
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import rich_click as click

from truss.cli.cannery.binary import resolve_cannery_binary  # noqa: F401
from truss.cli.cannery.errors import CanneryProtocolError  # noqa: F401
from truss.cli.cannery.runner import run_cannery as _run_cannery
from truss.cli.cli import truss_cli
from truss.cli.utils import common
from truss.cli.utils.output import console, console_to_stderr


def run_cannery(arguments):
    return _run_cannery(arguments)


def _output_option(function):
    return click.option(
        "--output",
        "output_format",
        type=click.Choice(["text", "json"]),
        default="text",
        show_default=True,
        help="Output format. Progress and status always go to stderr.",
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


@click.group()
def volume() -> None:
    """Manage Cannery-backed volumes."""


truss_cli.add_command(volume)


@volume.command(name="push")
@click.argument("path", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("ref", required=False)
@_output_option
@_clean_json_stdout
@common.common_options()
def push_volume(path: Path, ref: Optional[str], output_format: str) -> None:
    """Upload PATH as a volume, optionally naming it with REF."""
    arguments = ["push", str(path)]
    if ref is not None:
        arguments.append(ref)
    _emit_result(run_cannery(arguments), output_format)


@volume.command(name="ls")
@click.argument("namespace", required=False)
@click.option("--all", "include_all", is_flag=True, help="Include untagged versions.")
@_output_option
@_clean_json_stdout
@common.common_options()
def list_volumes(
    namespace: Optional[str], include_all: bool, output_format: str
) -> None:
    """List namespaces, or volumes in NAMESPACE."""
    arguments = ["ls"]
    if namespace is not None:
        arguments.append(namespace)
    if include_all:
        arguments.append("--all")
    _emit_result(run_cannery(arguments), output_format)


@volume.command(name="show")
@click.argument("ref")
@_output_option
@_clean_json_stdout
@common.common_options()
def show_volume(ref: str, output_format: str) -> None:
    """Show metadata and files for REF."""
    _emit_result(run_cannery(["show", ref]), output_format)


@volume.command(name="pull")
@click.argument("ref")
@click.argument("out_dir", type=click.Path(file_okay=False, path_type=Path))
@click.option(
    "--discard",
    is_flag=True,
    help="Download and verify content without writing files (benchmark mode).",
)
@_output_option
@_clean_json_stdout
@common.common_options()
def pull_volume(ref: str, out_dir: Path, discard: bool, output_format: str) -> None:
    """Download all files from REF into OUT_DIR."""
    arguments = ["pull", ref, str(out_dir)]
    if discard:
        arguments.append("--discard")
    _emit_result(run_cannery(arguments), output_format)
