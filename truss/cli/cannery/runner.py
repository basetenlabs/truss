from __future__ import annotations

import os
import platform
import signal
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TextIO, Tuple

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
    CanneryCancelled,
    CanneryClickException,
    CanneryProtocolError,
    CanneryUsageError,
    attach_failure_context,
    command_failure,
    error_category,
    safe_machine_identifier,
)
from truss.cli.cannery.progress import ProgressRenderer
from truss.cli.cannery.protocol import (
    CanneryProtocolConsumer,
    Phase0ProtocolConsumer,
    V1ProtocolConsumer,
    parse_protocol_bootstrap,
)

_CANCEL_GRACE_SECONDS = 5
_BOOTSTRAP_OUTPUT_LIMIT = 64 * 1024
_PHASE_0_ENVIRONMENT_VARIABLE = "TRUSS_CANNERY_PHASE0"
_SAFE_EXCEPTION_CLASS_NAMES = {
    OSError: "OSError",
    RuntimeError: "RuntimeError",
    TypeError: "TypeError",
    ValueError: "ValueError",
}


class _BoundedCapture:
    def __init__(self, stream: TextIO, limit: int) -> None:
        self._stream = stream
        self._limit = limit
        self.value = ""
        self.truncated = False

    def drain(self) -> None:
        for chunk in iter(lambda: self._stream.read(8192), ""):
            remaining = self._limit - len(self.value)
            if remaining > 0:
                self.value += chunk[:remaining]
            if len(chunk) > remaining:
                self.truncated = True


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


def _capture_bootstrap(
    binary: str, environment: Dict[str, str]
) -> Tuple[str, str, int]:
    popen_options: Dict[str, Any] = {}
    if os.name == "nt":
        popen_options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    try:
        process = subprocess.Popen(
            [binary, "protocol"],
            env=environment,
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
        raise CanneryClickException(
            "Failed to start the Cannery protocol bootstrap."
        ) from None
    if process.stdout is None or process.stderr is None:
        _cancel_process(process)
        raise CanneryClickException(
            "Failed to capture Cannery protocol bootstrap output."
        )

    stdout = _BoundedCapture(process.stdout, _BOOTSTRAP_OUTPUT_LIMIT)
    stderr = _BoundedCapture(process.stderr, _BOOTSTRAP_OUTPUT_LIMIT)
    threads = [
        threading.Thread(
            target=stdout.drain, name="truss-cannery-bootstrap-stdout", daemon=True
        ),
        threading.Thread(
            target=stderr.drain, name="truss-cannery-bootstrap-stderr", daemon=True
        ),
    ]
    for thread in threads:
        thread.start()
    try:
        return_code = process.wait()
    except KeyboardInterrupt:
        _cancel_process(process)
        raise
    finally:
        if process.poll() is None:
            _cancel_process(process)
        for thread in threads:
            thread.join()
    if stdout.truncated:
        raise CanneryProtocolError(
            "Cannery protocol bootstrap stdout exceeds the size limit."
        )
    return (stdout.value, stderr.value, return_code)


def run_cannery(
    arguments: List[str],
    remote: Optional[str] = None,
    protocol_consumer: Optional[CanneryProtocolConsumer] = None,
    binary_resolver: Optional[Callable[[], str]] = None,
    config_resolver: Callable[[Optional[str]], CanneryConfig] = resolve_cannery_config,
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
            remote=remote,
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
        raise CanneryCancelled() from None
    except CanneryCancelled:
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


def _default_protocol_consumer(
    arguments: List[str], config: CanneryConfig
) -> CanneryProtocolConsumer:
    if not arguments:
        raise CanneryUsageError("A Cannery operation is required.")
    phase_0 = os.environ.get(_PHASE_0_ENVIRONMENT_VARIABLE)
    if phase_0 is not None and phase_0 != "1":
        raise CanneryUsageError(
            f"{_PHASE_0_ENVIRONMENT_VARIABLE} must be 1 when explicitly enabled."
        )
    if phase_0 == "1":
        if not config.is_loopback:
            raise CanneryUsageError(
                "The Phase 0 Cannery protocol cannot be used with a non-loopback "
                "endpoint."
            )
        return Phase0ProtocolConsumer()
    return V1ProtocolConsumer(arguments[0])


def _run_cannery(
    *,
    arguments: List[str],
    correlation_id: str,
    diagnostic: DiagnosticLog,
    protocol_consumer: Optional[CanneryProtocolConsumer],
    binary_resolver: Optional[Callable[[], str]],
    config_resolver: Callable[[Optional[str]], CanneryConfig],
    auth_provider: Optional[CanneryAuthProvider],
    exchange_adapter: Optional[BasetenExchangeAdapter],
    remote: Optional[str],
) -> Dict[str, Any]:
    config = config_resolver(remote)
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
                binary = resolve_cannery_binary(
                    allow_path_fallback=config.allow_path_fallback
                )
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
        environment = child_environment(
            credential, config.org, correlation_id, diagnostic.path
        )
        consumer = protocol_consumer or _default_protocol_consumer(arguments, config)
        phase_0 = isinstance(consumer, Phase0ProtocolConsumer)
        if phase_0 and not config.is_loopback:
            raise CanneryUsageError(
                "The Phase 0 Cannery protocol is restricted to explicit loopback "
                "development. Non-loopback endpoints require machine protocol v1."
            )
        if phase_0 and artifact_version != "development-override":
            raise CanneryUsageError(
                "Pinned Cannery artifacts require machine protocol v1."
            )
        if phase_0:
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
        else:
            try:
                bootstrap_stdout, bootstrap_stderr, bootstrap_return_code = (
                    _capture_bootstrap(binary, environment)
                )
            except KeyboardInterrupt:
                raise CanneryCancelled() from None
            if bootstrap_stderr:
                diagnostic.record("bootstrap_stderr", message=bootstrap_stderr)
            bootstrap = parse_protocol_bootstrap(
                bootstrap_stdout, bootstrap_return_code
            )
            diagnostic.record(
                "protocol_negotiated",
                protocol_version=1,
                artifact_version=bootstrap.cannery_version,
            )
            argv = [binary, "--machine-protocol", "1", "--api", config.api, *arguments]

        popen_options: Dict[str, Any] = {}
        if os.name == "nt":
            popen_options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        try:
            process = subprocess.Popen(
                argv,
                env=environment,
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

        progress_renderer = ProgressRenderer()
        session = consumer.start(process.stdout, process.stderr, progress_renderer)
        interrupted = False
        terminal_exit_timed_out = False
        read_protocol_error: Optional[CanneryProtocolError] = None
        return_code: Optional[int] = None
        try:
            result = session.read_result()
            terminal_exit_timeout_sec = session.terminal_exit_timeout_sec
            try:
                return_code = process.wait(timeout=terminal_exit_timeout_sec)
            except subprocess.TimeoutExpired:
                terminal_exit_timed_out = True
                _cancel_process(process)
                return_code = process.poll()
                if return_code is None:
                    return_code = -1
        except CanneryProtocolError as exc:
            read_protocol_error = exc
            raise
        except KeyboardInterrupt:
            interrupted = True
            _cancel_process(process)
            cancelled_return_code = process.poll()
            if cancelled_return_code is None:
                cancelled_return_code = 130
            try:
                session.finish(cancelled_return_code, enforce_exit_status=False)
            except (CanneryProtocolError, KeyboardInterrupt):
                pass
            if session.stderr_diagnostic:
                diagnostic.record(
                    "subprocess_stderr", message=session.stderr_diagnostic
                )
            raise CanneryCancelled()
        finally:
            try:
                if process.poll() is None:
                    _cancel_process(process)
                if not interrupted:
                    completed_return_code = return_code
                    if completed_return_code is None:
                        completed_return_code = process.poll()
                    if completed_return_code is None:
                        completed_return_code = process.wait()
                    try:
                        try:
                            session.finish(
                                completed_return_code,
                                enforce_exit_status=not terminal_exit_timed_out,
                            )
                        except CanneryProtocolError:
                            if read_protocol_error is None:
                                raise
                    finally:
                        if session.stderr_diagnostic:
                            diagnostic.record(
                                "subprocess_stderr", message=session.stderr_diagnostic
                            )
            finally:
                progress_renderer.close()

        if terminal_exit_timed_out:
            raise CanneryProtocolError(
                "Cannery emitted a terminal machine record but did not exit within "
                f"{session.terminal_exit_timeout_sec:g} seconds."
            )
        if session.cancelled:
            raise CanneryCancelled()
        assert return_code is not None
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
        credential.raise_refresh_error()
        return result


def _safe_exception_class_name(error: BaseException) -> str:
    return _SAFE_EXCEPTION_CLASS_NAMES.get(type(error), "Exception")
