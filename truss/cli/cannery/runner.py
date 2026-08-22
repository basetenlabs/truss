from __future__ import annotations

import signal
import subprocess
from typing import Any, Callable, Dict, List, Mapping, Optional

import rich_click as click

from truss.cli.cannery.auth import child_environment, resolve_token_file
from truss.cli.cannery.binary import resolve_cannery_binary
from truss.cli.cannery.config import resolve_cannery_config
from truss.cli.cannery.protocol import CanneryProtocolConsumer, Phase0ProtocolConsumer

_CANCEL_GRACE_SECONDS = 5


def _format_machine_error(
    error: Optional[Mapping[str, Any]], return_code: int
) -> str:
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


def run_cannery(
    arguments: List[str],
    protocol_consumer: Optional[CanneryProtocolConsumer] = None,
    binary_resolver: Callable[[], str] = resolve_cannery_binary,
) -> Dict[str, Any]:
    config = resolve_cannery_config()
    token_file = resolve_token_file(config)
    binary = binary_resolver()
    argv = [
        binary,
        "-o",
        "json",
        "--progress",
        "machine",
        "--api",
        config.api,
        *arguments,
    ]

    try:
        process = subprocess.Popen(
            argv,
            env=child_environment(token_file, config.org),
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

    consumer = protocol_consumer or Phase0ProtocolConsumer()
    session = consumer.start(
        process.stdout,
        process.stderr,
        lambda message: click.echo(message, err=True),
    )
    try:
        result = session.read_result()
        return_code = process.wait()
    except KeyboardInterrupt:
        _cancel_process(process)
        raise click.Abort()
    finally:
        if process.poll() is None:
            _cancel_process(process)
        session.finish()

    machine_error = session.terminal_error
    if return_code == 2:
        raise click.UsageError(_format_machine_error(machine_error, return_code))
    if return_code != 0:
        raise click.ClickException(_format_machine_error(machine_error, return_code))
    if machine_error is not None:
        raise click.ClickException(_format_machine_error(machine_error, return_code))
    return result
