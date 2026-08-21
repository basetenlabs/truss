from __future__ import annotations

import ipaddress
import json
import os
import shutil
import signal
import subprocess
import threading
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional
from urllib.parse import urlparse

import rich_click as click

from truss.cli.cli import truss_cli
from truss.cli.utils import common
from truss.cli.utils.output import console, console_to_stderr

_DEFAULT_API = "http://127.0.0.1:8787"
_DEFAULT_ORG = "dev"
_PROTOCOL_VERSION = 1
_CANCEL_GRACE_SECONDS = 5


class CanneryProtocolError(click.ClickException):
    pass


def resolve_cannery_binary() -> str:
    configured = os.environ.get("TRUSS_CANNERY_BIN")
    candidate = configured or "cannery"
    resolved = shutil.which(candidate)
    if resolved is not None:
        return resolved

    if configured:
        raise click.ClickException(
            f"Cannot execute Cannery binary configured by TRUSS_CANNERY_BIN: "
            f"{configured!r}. Set it to an executable Cannery client."
        )
    raise click.ClickException(
        "Cannery client binary was not found on PATH. Install or build `cannery`, "
        "or set TRUSS_CANNERY_BIN to its executable path."
    )


def _is_loopback_endpoint(api: str) -> bool:
    parsed = urlparse(api)
    if parsed.scheme not in {"http", "https"} or parsed.hostname is None:
        raise click.UsageError(
            "TRUSS_CANNERY_API must be an http(s) URL with a hostname."
        )

    hostname = parsed.hostname.rstrip(".").lower()
    if hostname == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def _resolve_token_file(api: str) -> Optional[Path]:
    configured = os.environ.get("TRUSS_CANNERY_AUTH_TOKEN_FILE")
    if configured:
        token_file = Path(configured).expanduser()
        if not token_file.is_file():
            raise click.UsageError(
                "TRUSS_CANNERY_AUTH_TOKEN_FILE must point to an existing token "
                f"file; got {configured!r}."
            )
        return token_file.resolve()

    if _is_loopback_endpoint(api):
        return None

    # TODO(RUN-869): exchange the configured Baseten credential for a Cannery
    # token. Until that endpoint exists, remote use must be explicitly supplied
    # a token file instead of silently attempting unauthenticated access.
    raise click.UsageError(
        "A Cannery token is required for non-loopback endpoints. "
        "Set TRUSS_CANNERY_AUTH_TOKEN_FILE to an existing token file. "
        "Automatic Baseten-to-Cannery token exchange is pending RUN-869."
    )


def _child_environment(token_file: Optional[Path], org: str) -> Dict[str, str]:
    env = os.environ.copy()
    env.pop("CANNERY_AUTH_TOKEN_FILE", None)
    env["CANNERY_ORG"] = org
    if token_file is not None:
        env["CANNERY_AUTH_TOKEN_FILE"] = str(token_file)
    return env


def _event_kind(event: Mapping[str, Any]) -> Optional[str]:
    for key in ("type", "event", "kind"):
        value = event.get(key)
        if isinstance(value, str):
            return value
    return None


def _event_values(event: Mapping[str, Any]) -> Mapping[str, Any]:
    progress = event.get("progress")
    if isinstance(progress, dict):
        return {**event, **progress}
    return event


def _format_progress_event(event: Mapping[str, Any]) -> Optional[str]:
    values = _event_values(event)
    operation = values.get("operation") or values.get("command") or "transfer"
    phase = values.get("phase")
    label = f"Cannery {operation}"
    if isinstance(phase, str):
        label += f" ({phase})"

    counters = []
    for noun in ("files", "chunks", "bytes"):
        done = values.get(f"{noun}_done")
        if done is None:
            done = values.get(f"completed_{noun}")
        total = values.get(f"{noun}_total")
        if total is None:
            total = values.get(f"total_{noun}")
        if done is not None and total is not None:
            counters.append(f"{done}/{total} {noun}")
        elif done is not None:
            counters.append(f"{done} {noun}")

    if not counters:
        completed = values.get("completed")
        total = values.get("total")
        unit = values.get("unit") or "items"
        if completed is not None and total is not None:
            counters.append(f"{completed}/{total} {unit}")
        elif completed is not None:
            counters.append(f"{completed} {unit}")

    if not counters:
        return None
    return f"{label}: {', '.join(counters)}"


def _format_status_event(event: Mapping[str, Any]) -> Optional[str]:
    operation = event.get("operation") or event.get("command")
    phase = event.get("phase")
    status = event.get("status") or event.get("state")
    typed_values = [
        value for value in (operation, phase, status) if isinstance(value, str)
    ]
    if not typed_values:
        return None
    rendered = f"Cannery {' — '.join(typed_values)}"
    message = event.get("message")
    if isinstance(message, str) and message:
        rendered += f": {message}"
    return rendered


def _render_machine_event(event: Mapping[str, Any]) -> None:
    kind = _event_kind(event)
    rendered = None
    if kind == "progress":
        rendered = _format_progress_event(event)
    elif kind in {"started", "status", "warning"}:
        rendered = _format_status_event(event)
    if rendered:
        click.echo(rendered, err=True)


def _validate_machine_event(event: Any) -> Mapping[str, Any]:
    if not isinstance(event, dict):
        raise CanneryProtocolError(
            "Cannery machine progress emitted a JSON value that is not an object."
        )
    version = event.get("protocol_version")
    if type(version) is not int or version != _PROTOCOL_VERSION:
        raise CanneryProtocolError(
            "Unsupported Cannery machine progress protocol version "
            f"{version!r}; Truss requires version {_PROTOCOL_VERSION}."
        )
    if _event_kind(event) is None:
        raise CanneryProtocolError(
            "Cannery machine progress event is missing its event type."
        )
    return event


def _drain_machine_events(
    lines: Iterable[str], events: List[Mapping[str, Any]], errors: List[Exception]
) -> None:
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            event = _validate_machine_event(json.loads(line))
            events.append(event)
            _render_machine_event(event)
        except json.JSONDecodeError as exc:
            errors.append(
                CanneryProtocolError(
                    "Cannery machine progress emitted invalid NDJSON on stderr "
                    f"at line {line_number}: {exc.msg}."
                )
            )
        except CanneryProtocolError as exc:
            errors.append(exc)


def _machine_error(events: Iterable[Mapping[str, Any]]) -> Optional[Mapping[str, Any]]:
    for event in reversed(list(events)):
        if _event_kind(event) == "error":
            error = event.get("error")
            if isinstance(error, dict):
                return {**event, **error}
            return event
    return None


def _format_machine_error(error: Optional[Mapping[str, Any]], return_code: int) -> str:
    if error is None:
        return (
            f"Cannery exited with status {return_code} without a machine error event."
        )

    reason = error.get("reason") or error.get("status")
    message = error.get("message") or error.get("detail")
    hint = error.get("hint")
    parts = []
    if isinstance(reason, str):
        parts.append(reason)
    if isinstance(message, str) and message != reason:
        parts.append(message)
    rendered = ": ".join(parts) or f"Cannery exited with status {return_code}"
    if isinstance(hint, str):
        rendered += f" Hint: {hint}"
    return rendered


def _parse_result(stdout: str) -> Dict[str, Any]:
    decoder = json.JSONDecoder()
    stripped = stdout.lstrip()
    if not stripped:
        raise CanneryProtocolError(
            "Cannery succeeded without emitting its final JSON result."
        )
    try:
        result, end = decoder.raw_decode(stripped)
    except json.JSONDecodeError as exc:
        raise CanneryProtocolError(
            f"Cannery emitted an invalid final JSON result: {exc.msg}."
        ) from exc
    if stripped[end:].strip():
        raise CanneryProtocolError(
            "Cannery emitted more than one value on result stdout."
        )
    if not isinstance(result, dict):
        raise CanneryProtocolError("Cannery final JSON result must be an object.")
    version = result.get("protocol_version")
    if type(version) is not int or version != _PROTOCOL_VERSION:
        raise CanneryProtocolError(
            "Unsupported Cannery result protocol version "
            f"{version!r}; Truss requires version {_PROTOCOL_VERSION}."
        )
    return result


def _cancel_process(process: "subprocess.Popen[str]") -> None:
    if process.poll() is not None:
        return
    try:
        process.send_signal(signal.SIGINT)
        process.wait(timeout=_CANCEL_GRACE_SECONDS)
        return
    except (OSError, subprocess.TimeoutExpired):
        pass

    try:
        process.terminate()
        process.wait(timeout=2)
        return
    except (OSError, subprocess.TimeoutExpired):
        pass

    process.kill()
    process.wait()


def run_cannery(arguments: List[str]) -> Dict[str, Any]:
    api = os.environ.get("TRUSS_CANNERY_API", _DEFAULT_API)
    org = os.environ.get("TRUSS_CANNERY_ORG", _DEFAULT_ORG)
    token_file = _resolve_token_file(api)
    binary = resolve_cannery_binary()
    argv = [binary, "-o", "json", "--progress", "machine", "--api", api, *arguments]

    try:
        process = subprocess.Popen(
            argv,
            env=_child_environment(token_file, org),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
    except OSError as exc:
        raise click.ClickException(
            f"Failed to start Cannery binary {binary!r}: {exc}"
        ) from exc

    if process.stdout is None or process.stderr is None:
        _cancel_process(process)
        raise click.ClickException("Failed to capture Cannery subprocess output.")

    events: List[Mapping[str, Any]] = []
    protocol_errors: List[Exception] = []
    stderr_thread = threading.Thread(
        target=_drain_machine_events,
        args=(process.stderr, events, protocol_errors),
        name="truss-cannery-stderr",
        daemon=True,
    )
    stderr_thread.start()
    try:
        stdout = process.stdout.read()
        return_code = process.wait()
    except KeyboardInterrupt:
        _cancel_process(process)
        raise click.Abort()
    finally:
        if process.poll() is None:
            _cancel_process(process)
        stderr_thread.join()

    if protocol_errors:
        raise protocol_errors[0]

    machine_error = _machine_error(events)
    if return_code == 2:
        raise click.UsageError(_format_machine_error(machine_error, return_code))
    if return_code != 0:
        raise click.ClickException(_format_machine_error(machine_error, return_code))
    if machine_error is not None:
        raise click.ClickException(_format_machine_error(machine_error, return_code))
    return _parse_result(stdout)


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
