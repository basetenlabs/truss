from __future__ import annotations

import os
import platform
import signal
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import rich_click as click

from truss.cli.cannery.auth import (
    BasetenExchangeAdapter,
    CanneryAuthProvider,
    child_environment,
    select_auth_provider,
)
from truss.cli.cannery.binary import binary_diagnostic_metadata, resolve_cannery_binary
from truss.cli.cannery.config import CanneryConfig, resolve_cannery_config
from truss.cli.cannery.diagnostics import (
    DiagnosticLog,
    diagnostic_failure_suffix,
    endpoint_hostname,
)
from truss.cli.cannery.errors import (
    CanneryClickException,
    CanneryUsageError,
    attach_failure_context,
    command_failure,
    error_category,
    safe_machine_identifier,
)
from truss.cli.cannery.protocol import CanneryProtocolConsumer, Phase0ProtocolConsumer

_CANCEL_GRACE_SECONDS = 5
_SAFE_EXCEPTION_CLASS_NAMES = {
    OSError: "OSError",
    RuntimeError: "RuntimeError",
    TypeError: "TypeError",
    ValueError: "ValueError",
}


def _cancel_process(process: "subprocess.Popen[str]") -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "nt" and hasattr(signal, "CTRL_BREAK_EVENT"):
            process.send_signal(signal.CTRL_BREAK_EVENT)
        else:
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
    binary_resolver: Optional[Callable[[], str]] = None,
    config_resolver: Callable[[], CanneryConfig] = resolve_cannery_config,
    auth_provider: Optional[CanneryAuthProvider] = None,
    exchange_adapter: Optional[BasetenExchangeAdapter] = None,
) -> Dict[str, Any]:
    correlation_id = str(uuid.uuid4())
    diagnostic = DiagnosticLog.create(correlation_id)
    operation = arguments[0] if arguments else "<missing>"
    started_at = time.monotonic()
    diagnostic.record(
        "started",
        operation=operation,
        protocol_version=1,
        operating_system=f"{platform.system()}-{platform.machine()}",
    )

    try:
        result = _run_cannery(
            arguments=arguments,
            correlation_id=correlation_id,
            diagnostic=diagnostic,
            protocol_consumer=protocol_consumer,
            binary_resolver=binary_resolver,
            config_resolver=config_resolver,
            auth_provider=auth_provider,
            exchange_adapter=exchange_adapter,
        )
        result.setdefault("correlation_id", correlation_id)
        diagnostic.record(
            "completed", operation=operation, duration_sec=time.monotonic() - started_at
        )
    except click.Abort:
        diagnostic.record(
            "cancelled", operation=operation, duration_sec=time.monotonic() - started_at
        )
        click.echo(diagnostic_failure_suffix(correlation_id, diagnostic.path), err=True)
        raise
    except (CanneryClickException, CanneryUsageError) as exc:
        diagnostic.record(
            "failed",
            operation=operation,
            message="Cannery operation failed.",
            exception_class=_safe_exception_class_name(exc),
            duration_sec=time.monotonic() - started_at,
        )
        raise attach_failure_context(exc, correlation_id, diagnostic.path)
    except click.ClickException as exc:
        diagnostic.record(
            "failed",
            operation=operation,
            message="Cannery wrapper rejected an unreviewed error.",
            exception_class=_safe_exception_class_name(exc),
            duration_sec=time.monotonic() - started_at,
        )
        wrapped = CanneryClickException(
            "Cannery wrapper failed before a safe typed error was available."
        )
        raise attach_failure_context(wrapped, correlation_id, diagnostic.path) from None
    except Exception as exc:
        diagnostic.record(
            "failed",
            operation=operation,
            message="Cannery wrapper encountered an unexpected exception.",
            exception_class=_safe_exception_class_name(exc),
            duration_sec=time.monotonic() - started_at,
        )
        wrapped = CanneryClickException(
            "Cannery wrapper failed before a safe typed error was available."
        )
        raise attach_failure_context(wrapped, correlation_id, diagnostic.path) from None
    else:
        diagnostic.delete()
        return result


def _run_cannery(
    *,
    arguments: List[str],
    correlation_id: str,
    diagnostic: DiagnosticLog,
    protocol_consumer: Optional[CanneryProtocolConsumer],
    binary_resolver: Optional[Callable[[], str]],
    config_resolver: Callable[[], CanneryConfig],
    auth_provider: Optional[CanneryAuthProvider],
    exchange_adapter: Optional[BasetenExchangeAdapter],
) -> Dict[str, Any]:
    config = config_resolver()
    diagnostic.record(
        "configured",
        endpoint_hostname=endpoint_hostname(config.api),
        operation=arguments[0] if arguments else "<missing>",
    )
    provider = auth_provider or select_auth_provider(config, exchange_adapter)

    with provider.acquire(correlation_id) as credential:
        diagnostic.record("authenticated", mechanism=credential.mechanism)
        if binary_resolver is None:
            try:
                binary = resolve_cannery_binary(allow_path_fallback=config.is_loopback)
            except click.ClickException:
                raise CanneryClickException(
                    "Could not resolve a trusted Cannery binary. For local "
                    "development, set TRUSS_CANNERY_BIN to an executable client."
                ) from None
        else:
            binary = binary_resolver()
        artifact_version, artifact_sha256 = binary_diagnostic_metadata(Path(binary))
        diagnostic.record(
            "binary_resolved",
            binary_path=binary,
            artifact_version=artifact_version,
            artifact_sha256=artifact_sha256,
        )
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

        popen_options: Dict[str, Any] = {}
        if os.name == "nt":
            popen_options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        try:
            process = subprocess.Popen(
                argv,
                env=child_environment(
                    credential, config.org, correlation_id, diagnostic.path
                ),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                **popen_options,
            )
        except OSError:
            raise CanneryClickException("Failed to start the Cannery binary.") from None

        if process.stdout is None or process.stderr is None:
            _cancel_process(process)
            raise CanneryClickException("Failed to capture Cannery subprocess output.")

        consumer = protocol_consumer or Phase0ProtocolConsumer()
        session = consumer.start(
            process.stdout,
            process.stderr,
            lambda message: click.echo(message, err=True),
        )
        interrupted = False
        try:
            result = session.read_result()
            return_code = process.wait()
        except KeyboardInterrupt:
            interrupted = True
            _cancel_process(process)
            raise click.Abort()
        finally:
            if process.poll() is None:
                _cancel_process(process)
            if not interrupted:
                session.finish()

        machine_error = session.terminal_error
        if return_code != 0 or machine_error is not None:
            diagnostic.record(
                "subprocess_failed",
                category=error_category(machine_error, return_code).value,
                exit_code=return_code,
                phase=session.last_phase,
                reason=(
                    safe_machine_identifier(machine_error.get("reason"))
                    if machine_error
                    else None
                ),
                retryable=machine_error.get("retryable") if machine_error else None,
            )
            raise command_failure(
                machine_error, return_code, correlation_id, diagnostic.path
            )
        return result


def _safe_exception_class_name(error: BaseException) -> str:
    return _SAFE_EXCEPTION_CLASS_NAMES.get(type(error), "Exception")
