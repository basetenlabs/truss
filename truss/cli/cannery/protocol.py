from __future__ import annotations

import json
import threading
from typing import Any, BinaryIO, Callable, Dict, Mapping, Optional, Protocol

from truss.cli.cannery.diagnostics import redact_text
from truss.cli.cannery.errors import CanneryProtocolError
from truss.cli.cannery.progress import BoundedProgressState, event_kind, format_event
from truss.cli.cannery.v1_protocol import (
    V1_MACHINE_ENCODING,
    V1ProtocolConsumer,
    V1ProtoJSONProtocolConsumer,
    parse_protocol_bootstrap,
)

__all__ = [
    "CanneryProtocolConsumer",
    "CanneryProtocolSession",
    "Phase0ProtocolConsumer",
    "V1_MACHINE_ENCODING",
    "V1ProtocolConsumer",
    "V1ProtoJSONProtocolConsumer",
    "parse_protocol_bootstrap",
]

_PHASE_0_PROTOCOL_VERSION = 1
_MAX_STDERR_CHARACTERS = 64 * 1024


class CanneryProtocolSession(Protocol):
    """One running subprocess protocol session.

    Generated protocol v1 can replace this implementation without changing the
    command, authentication, binary-resolution, or subprocess-runner modules.
    """

    @property
    def terminal_error(self) -> Optional[Mapping[str, Any]]: ...

    @property
    def last_phase(self) -> Optional[str]: ...

    @property
    def cancelled(self) -> bool: ...

    @property
    def stderr_diagnostic(self) -> str: ...

    @property
    def terminal_exit_timeout_sec(self) -> Optional[float]: ...

    def read_result(self) -> Dict[str, Any]: ...

    def finish(self, return_code: int, *, enforce_exit_status: bool = True) -> None: ...


class CanneryProtocolConsumer(Protocol):
    def start(
        self, stdout: BinaryIO, stderr: BinaryIO, render_progress: Callable[[str], None]
    ) -> CanneryProtocolSession: ...


class Phase0ProtocolConsumer:
    """Consumes the temporary split stdout/stderr JSON protocol."""

    def start(
        self, stdout: BinaryIO, stderr: BinaryIO, render_progress: Callable[[str], None]
    ) -> CanneryProtocolSession:
        return _Phase0ProtocolSession(stdout, stderr, render_progress)


class _Phase0ProtocolSession:
    def __init__(
        self, stdout: BinaryIO, stderr: BinaryIO, render_progress: Callable[[str], None]
    ) -> None:
        self._stdout = stdout
        self._stderr = stderr
        self._render_progress = render_progress
        self._state = BoundedProgressState()
        self._protocol_error: Optional[CanneryProtocolError] = None
        self._stderr_diagnostic = ""
        self._stderr_truncated = False
        self._stderr_thread = threading.Thread(
            target=self._drain_machine_events, name="truss-cannery-stderr", daemon=True
        )
        self._stderr_thread.start()

    @property
    def terminal_error(self) -> Optional[Mapping[str, Any]]:
        return self._state.terminal_error

    @property
    def last_phase(self) -> Optional[str]:
        return self._state.last_phase

    @property
    def cancelled(self) -> bool:
        return False

    @property
    def stderr_diagnostic(self) -> str:
        prefix = "[earlier stderr truncated]\n" if self._stderr_truncated else ""
        return prefix + redact_text(self._stderr_diagnostic)

    @property
    def terminal_exit_timeout_sec(self) -> Optional[float]:
        return None

    def _drain_machine_events(self) -> None:
        for line_number, encoded_line in enumerate(self._stderr, start=1):
            if not isinstance(encoded_line, bytes):
                if self._protocol_error is None:
                    self._protocol_error = CanneryProtocolError(
                        "Cannery Phase 0 stderr is not a binary stream."
                    )
                return
            line = encoded_line.decode("utf-8", errors="replace")
            self._stderr_diagnostic += line
            if len(self._stderr_diagnostic) > _MAX_STDERR_CHARACTERS:
                self._stderr_truncated = True
                self._stderr_diagnostic = self._stderr_diagnostic[
                    -_MAX_STDERR_CHARACTERS:
                ]
            if not line.strip():
                continue
            try:
                event = _validate_machine_event(json.loads(line))
                self._state.observe(event)
                rendered = format_event(event)
                if rendered:
                    self._render_progress(rendered)
            except json.JSONDecodeError as exc:
                if self._protocol_error is None:
                    self._protocol_error = CanneryProtocolError(
                        "Cannery machine progress emitted invalid NDJSON on stderr "
                        f"at line {line_number}: {exc.msg}."
                    )
            except CanneryProtocolError as exc:
                if self._protocol_error is None:
                    self._protocol_error = exc

    def read_result(self) -> Dict[str, Any]:
        return _parse_result(self._stdout.read())

    def finish(self, return_code: int, *, enforce_exit_status: bool = True) -> None:
        self._stderr_thread.join()
        if self._protocol_error is not None:
            raise self._protocol_error


def _validate_machine_event(event: Any) -> Mapping[str, Any]:
    if not isinstance(event, dict):
        raise CanneryProtocolError(
            "Cannery machine progress emitted a JSON value that is not an object."
        )
    version = event.get("protocol_version")
    if type(version) is not int or version != _PHASE_0_PROTOCOL_VERSION:
        raise CanneryProtocolError(
            "Unsupported Cannery machine progress protocol version; "
            f"Truss requires version {_PHASE_0_PROTOCOL_VERSION}."
        )
    if event_kind(event) is None:
        raise CanneryProtocolError(
            "Cannery machine progress event is missing its event type."
        )
    return event


def _parse_result(stdout: bytes) -> Dict[str, Any]:
    if not isinstance(stdout, bytes):
        raise CanneryProtocolError("Cannery Phase 0 stdout is not a binary stream.")
    try:
        decoded_stdout = stdout.decode("utf-8")
    except UnicodeDecodeError:
        raise CanneryProtocolError(
            "Cannery emitted an invalid UTF-8 final JSON result."
        ) from None
    decoder = json.JSONDecoder()
    stripped = decoded_stdout.lstrip()
    if not stripped:
        raise CanneryProtocolError(
            "Cannery succeeded without emitting its final JSON result."
        )
    try:
        result, end = decoder.raw_decode(stripped)
    except json.JSONDecodeError as exc:
        raise CanneryProtocolError(
            f"Cannery emitted an invalid final JSON result: {exc.msg}."
        ) from None
    if stripped[end:].strip():
        raise CanneryProtocolError(
            "Cannery emitted more than one value on result stdout."
        )
    if not isinstance(result, dict):
        raise CanneryProtocolError("Cannery final JSON result must be an object.")
    version = result.get("protocol_version")
    if type(version) is not int or version != _PHASE_0_PROTOCOL_VERSION:
        raise CanneryProtocolError(
            "Unsupported Cannery result protocol version; "
            f"Truss requires version {_PHASE_0_PROTOCOL_VERSION}."
        )
    return result
