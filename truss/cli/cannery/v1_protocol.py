from __future__ import annotations

import codecs
import json
import math
import re
import threading
import time
from collections import deque
from typing import (
    Any,
    BinaryIO,
    Callable,
    Deque,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Protocol,
)

from google.protobuf import json_format
from google.protobuf.message import DecodeError, Message

from truss.cli.cannery.diagnostics import redact_text
from truss.cli.cannery.errors import CanneryProtocolError, TypedMachineError
from truss.cli.cannery.generated import cannery_cli_v1_pb2 as protocol_v1

_PROTOCOL_VERSION = 1
V1_MACHINE_ENCODING = "protobuf-delimited"
_MAX_RECORD_BYTES = 8 * 1024 * 1024
_MAX_VARINT_BYTES = 4
_MAX_STDERR_CHARACTERS = 64 * 1024
_STDERR_REDACTION_CONTEXT_CHARACTERS = 1024
_MAX_PAGE_ENTRIES = 1_000
_TERMINAL_EXIT_TIMEOUT_SEC = 5.0
_STREAM_DRAIN_TIMEOUT_SEC = 5.0
_STABLE_REASON = re.compile(r"^[A-Z][A-Z0-9_]{0,127}$")
_REQUIRED_PULL_SELECTED_TOTALS = {
    "logicalBytes": "logical_bytes",
    "fileCount": "file_count",
    "directoryCount": "directory_count",
}

_OPERATION_BY_COMMAND = {
    "push": protocol_v1.OPERATION_PUSH,
    "ls": protocol_v1.OPERATION_LIST,
    "show": protocol_v1.OPERATION_SHOW,
    "pull": protocol_v1.OPERATION_PULL,
}
_KNOWN_OPERATIONS = frozenset(_OPERATION_BY_COMMAND.values())
_REQUEST_BY_COMMAND = {"push": "push", "ls": "list", "show": "show", "pull": "pull"}
_OPERATION_WIRE_NAME = {
    "push": "OPERATION_PUSH",
    "ls": "OPERATION_LIST",
    "show": "OPERATION_SHOW",
    "pull": "OPERATION_PULL",
}
_DISPLAY_OPERATION = {"ls": "list", "push": "push", "show": "show", "pull": "pull"}
_PHASES_BY_COMMAND = {
    "push": frozenset({"scan", "hash", "upload", "commit"}),
    "ls": frozenset({"list"}),
    "show": frozenset({"resolve", "inspect"}),
    "pull": frozenset({"resolve", "validate", "download", "verify", "publish"}),
}
_ERROR_CATEGORY = {
    protocol_v1.ERROR_CATEGORY_INVALID_ARGUMENT: "usage",
    protocol_v1.ERROR_CATEGORY_AUTHENTICATION: "authentication",
    protocol_v1.ERROR_CATEGORY_AUTHORIZATION: "authorization",
    protocol_v1.ERROR_CATEGORY_NOT_FOUND: "not_found",
    protocol_v1.ERROR_CATEGORY_CONFLICT: "conflict",
    protocol_v1.ERROR_CATEGORY_THROTTLED: "throttled",
    protocol_v1.ERROR_CATEGORY_QUOTA: "quota",
    protocol_v1.ERROR_CATEGORY_UNAVAILABLE: "network",
    protocol_v1.ERROR_CATEGORY_INTEGRITY: "integrity",
    protocol_v1.ERROR_CATEGORY_UNSUPPORTED_PROTOCOL: "incompatible_client",
    protocol_v1.ERROR_CATEGORY_INTERNAL: "server",
}
_DETAIL_BY_CATEGORY = {
    protocol_v1.ERROR_CATEGORY_INVALID_ARGUMENT: "invalid_argument",
    protocol_v1.ERROR_CATEGORY_AUTHENTICATION: "authentication",
    protocol_v1.ERROR_CATEGORY_AUTHORIZATION: "authorization",
    protocol_v1.ERROR_CATEGORY_NOT_FOUND: "not_found",
    protocol_v1.ERROR_CATEGORY_CONFLICT: "conflict",
    protocol_v1.ERROR_CATEGORY_THROTTLED: "throttled",
    protocol_v1.ERROR_CATEGORY_QUOTA: "quota",
    protocol_v1.ERROR_CATEGORY_UNAVAILABLE: "unavailable",
    protocol_v1.ERROR_CATEGORY_INTEGRITY: "integrity",
    protocol_v1.ERROR_CATEGORY_UNSUPPORTED_PROTOCOL: "unsupported_protocol",
}
_KNOWN_REFERENCE_ENTRY_KINDS = frozenset(
    {
        protocol_v1.REFERENCE_ENTRY_KIND_TAG,
        protocol_v1.REFERENCE_ENTRY_KIND_IMMUTABLE_DIGEST,
    }
)
_KNOWN_FILE_ENTRY_KINDS = frozenset(
    {
        protocol_v1.FILE_ENTRY_KIND_FILE,
        protocol_v1.FILE_ENTRY_KIND_DIRECTORY,
        protocol_v1.FILE_ENTRY_KIND_SYMLINK,
    }
)


def parse_protocol_bootstrap(
    stdout: str, return_code: int
) -> protocol_v1.ProtocolBootstrapV1:
    if return_code != 0:
        raise CanneryProtocolError(
            f"Cannery protocol bootstrap exited with status {return_code}."
        )
    lines = stdout.splitlines(keepends=True)
    if len(lines) != 1 or not lines[0].endswith("\n"):
        raise CanneryProtocolError(
            "Cannery protocol bootstrap must emit exactly one newline-terminated object."
        )
    try:
        document = json.loads(lines[0])
    except json.JSONDecodeError as exc:
        raise CanneryProtocolError(
            f"Cannery protocol bootstrap emitted invalid JSON: {exc.msg}."
        ) from None
    if not isinstance(document, dict):
        raise CanneryProtocolError(
            "Cannery protocol bootstrap must emit a JSON object."
        )
    _validate_bootstrap_wire_types(document)
    bootstrap = protocol_v1.ProtocolBootstrapV1()
    try:
        json_format.ParseDict(document, bootstrap)
    except json_format.ParseError as exc:
        raise CanneryProtocolError(
            f"Cannery protocol bootstrap does not match its schema: {exc}."
        ) from None
    if bootstrap.bootstrap_version != 1:
        raise CanneryProtocolError(
            "Unsupported Cannery protocol bootstrap version; Truss requires version 1."
        )
    if not bootstrap.cannery_version:
        raise CanneryProtocolError(
            "Cannery protocol bootstrap omitted the binary version."
        )
    if _PROTOCOL_VERSION not in bootstrap.supported_machine_protocols:
        raise CanneryProtocolError(
            "Cannery does not support machine protocol version 1."
        )
    if V1_MACHINE_ENCODING not in bootstrap.supported_encodings:
        raise CanneryProtocolError(
            "Cannery does not support the protobuf-delimited v1 encoding."
        )
    return bootstrap


def _validate_bootstrap_wire_types(document: Mapping[str, Any]) -> None:
    required = {
        "bootstrap_version",
        "cannery_version",
        "supported_machine_protocols",
        "supported_encodings",
    }
    if set(document) != required:
        raise CanneryProtocolError(
            "Cannery protocol bootstrap fields do not match bootstrap version 1."
        )
    if type(document["bootstrap_version"]) is not int:
        raise CanneryProtocolError(
            "Cannery protocol bootstrap version must be an integer."
        )
    if not isinstance(document["cannery_version"], str):
        raise CanneryProtocolError(
            "Cannery protocol bootstrap binary version must be a string."
        )
    protocols = document["supported_machine_protocols"]
    encodings = document["supported_encodings"]
    if not isinstance(protocols, list) or any(
        type(value) is not int for value in protocols
    ):
        raise CanneryProtocolError(
            "Cannery protocol bootstrap machine protocols must be integers."
        )
    if not isinstance(encodings, list) or any(
        not isinstance(value, str) for value in encodings
    ):
        raise CanneryProtocolError(
            "Cannery protocol bootstrap encodings must be strings."
        )


class V1ProtocolConsumer:
    def __init__(self, command: str) -> None:
        if command not in _OPERATION_BY_COMMAND:
            raise CanneryProtocolError(f"Unsupported Cannery operation: {command}.")
        self._command = command

    def start(
        self, stdout: BinaryIO, stderr: BinaryIO, render_progress: Callable[[str], None]
    ) -> "_V1ProtocolSession":
        return _V1ProtocolSession(
            self._command, _DelimitedRecordReader(stdout), stderr, render_progress
        )


class V1ProtoJSONProtocolConsumer:
    """Explicit local/debug consumer for the optional ProtoJSON encoding."""

    def __init__(self, command: str) -> None:
        if command not in _OPERATION_BY_COMMAND:
            raise CanneryProtocolError(f"Unsupported Cannery operation: {command}.")
        self._command = command

    def start(
        self, stdout: BinaryIO, stderr: BinaryIO, render_progress: Callable[[str], None]
    ) -> "_V1ProtocolSession":
        return _V1ProtocolSession(
            self._command, _ProtoJSONRecordReader(stdout), stderr, render_progress
        )


class _DelimitedRecordReader:
    def __init__(self, stdout: BinaryIO) -> None:
        self._stdout = stdout
        self._next_frame_number = 1

    def read(self) -> Optional[protocol_v1.MachineRecordV1]:
        frame_number = self._next_frame_number
        payload_size = _read_canonical_varint(self._stdout, frame_number)
        if payload_size is None:
            return None
        payload = _read_exact_payload(self._stdout, payload_size, frame_number)
        record = protocol_v1.MachineRecordV1()
        try:
            record.ParseFromString(payload)
        except DecodeError:
            raise CanneryProtocolError(
                f"Cannery machine frame {frame_number} contains malformed Protobuf."
            ) from None
        if not record.ListFields():
            raise CanneryProtocolError(
                f"Cannery machine frame {frame_number} decoded to the default record."
            )
        self._next_frame_number += 1
        return record


class _RecordReader(Protocol):
    def read(self) -> Optional[protocol_v1.MachineRecordV1]: ...


class _ProtoJSONRecordReader:
    def __init__(self, stdout: BinaryIO) -> None:
        self._stdout = stdout
        self._line_number = 0

    def read(self) -> Optional[protocol_v1.MachineRecordV1]:
        line = self._stdout.readline(_MAX_RECORD_BYTES + 2)
        if not line:
            return None
        self._line_number += 1
        line_number = self._line_number
        if not isinstance(line, bytes):
            raise CanneryProtocolError(
                "Cannery ProtoJSON debug stdout must be a binary pipe."
            )
        if len(line) > _MAX_RECORD_BYTES + 1:
            raise CanneryProtocolError(
                f"Cannery machine record at line {line_number} exceeds the size limit."
            )
        if not line.endswith(b"\n"):
            raise CanneryProtocolError(
                f"Cannery machine record at line {line_number} is not newline-terminated."
            )
        if not line.strip():
            raise CanneryProtocolError(
                f"Cannery machine stream contains an empty record at line {line_number}."
            )
        try:
            text = line.decode("utf-8")
        except UnicodeDecodeError:
            raise CanneryProtocolError(
                f"Cannery emitted invalid UTF-8 ProtoJSON at line {line_number}."
            ) from None
        try:
            document = json.loads(text)
        except json.JSONDecodeError as exc:
            raise CanneryProtocolError(
                f"Cannery emitted invalid ProtoJSON at line {line_number}: {exc.msg}."
            ) from None
        if not isinstance(document, dict):
            raise CanneryProtocolError(
                f"Cannery machine record at line {line_number} is not an object."
            )
        _validate_required_pull_selected_totals(document)
        _validate_symbolic_enums(document)
        record = protocol_v1.MachineRecordV1()
        try:
            json_format.ParseDict(document, record)
        except json_format.ParseError as exc:
            raise CanneryProtocolError(
                f"Cannery machine record at line {line_number} does not match v1: {exc}."
            ) from None
        return record


def _read_canonical_varint(stdout: BinaryIO, frame_number: int) -> Optional[int]:
    value = 0
    for byte_index in range(_MAX_VARINT_BYTES):
        encoded = stdout.read(1)
        if not encoded:
            if byte_index == 0:
                return None
            raise CanneryProtocolError(
                f"Cannery machine frame {frame_number} has a truncated length prefix."
            )
        if not isinstance(encoded, bytes) or len(encoded) != 1:
            raise CanneryProtocolError(
                "Cannery machine stdout did not provide a binary byte stream."
            )
        byte = encoded[0]
        value |= (byte & 0x7F) << (7 * byte_index)
        if byte & 0x80:
            if byte_index == _MAX_VARINT_BYTES - 1:
                raise CanneryProtocolError(
                    f"Cannery machine frame {frame_number} has an overlong length prefix."
                )
            continue
        if byte_index > 0 and value < 1 << (7 * byte_index):
            raise CanneryProtocolError(
                f"Cannery machine frame {frame_number} has a non-minimal length prefix."
            )
        if value == 0:
            raise CanneryProtocolError(
                f"Cannery machine frame {frame_number} has a zero-length payload."
            )
        if value > _MAX_RECORD_BYTES:
            raise CanneryProtocolError(
                f"Cannery machine frame {frame_number} exceeds the size limit."
            )
        return value
    raise AssertionError("canonical varint reader exhausted without a decision")


def _read_exact_payload(
    stdout: BinaryIO, payload_size: int, frame_number: int
) -> bytearray:
    payload = bytearray(payload_size)
    offset = 0
    while offset < payload_size:
        chunk = stdout.read(payload_size - offset)
        if not chunk:
            raise CanneryProtocolError(
                f"Cannery machine frame {frame_number} has a truncated payload."
            )
        if not isinstance(chunk, bytes) or len(chunk) > payload_size - offset:
            raise CanneryProtocolError(
                "Cannery machine stdout did not provide a bounded binary byte stream."
            )
        payload[offset : offset + len(chunk)] = chunk
        offset += len(chunk)
    return payload


class _V1ProtocolSession:
    def __init__(
        self,
        command: str,
        record_reader: _RecordReader,
        stderr: BinaryIO,
        render_progress: Callable[[str], None],
    ) -> None:
        self._command = command
        self._record_reader = record_reader
        self._render_progress = render_progress
        self._operation_id: Optional[str] = None
        self._next_sequence = 1
        self._record_count = 0
        self._terminal_kind: Optional[str] = None
        self._terminal_result: Optional[Dict[str, Any]] = None
        self._terminal_error: Optional[Mapping[str, Any]] = None
        self._last_phase: Optional[str] = None
        self._progress_counters: Dict[str, int] = {}
        self._stream_complete = False
        self._stderr = _BoundedStderrCapture(stderr)
        self._stderr_thread = threading.Thread(
            target=self._stderr.drain, name="truss-cannery-v1-stderr", daemon=True
        )
        self._stderr_thread.start()

    @property
    def terminal_error(self) -> Optional[Mapping[str, Any]]:
        return self._terminal_error

    @property
    def last_phase(self) -> Optional[str]:
        return self._last_phase

    @property
    def cancelled(self) -> bool:
        return self._terminal_kind == "cancelled"

    @property
    def stderr_diagnostic(self) -> str:
        return self._stderr.value

    @property
    def terminal_exit_timeout_sec(self) -> Optional[float]:
        return _TERMINAL_EXIT_TIMEOUT_SEC

    def read_result(self) -> Dict[str, Any]:
        while self._terminal_kind is None:
            record = self._record_reader.read()
            if record is None:
                self._stream_complete = True
                break
            self._observe(record)
        return self._terminal_result or {}

    def _observe(self, record: protocol_v1.MachineRecordV1) -> None:
        if self._terminal_kind is not None:
            raise CanneryProtocolError(
                "Cannery emitted a record after its terminal machine record."
            )
        payload = record.WhichOneof("payload")
        if record.protocol_version != _PROTOCOL_VERSION:
            raise CanneryProtocolError(
                "Unsupported Cannery machine protocol version; Truss requires version 1."
            )
        if record.sequence != self._next_sequence:
            raise CanneryProtocolError(
                "Cannery machine record sequence is not contiguous; "
                f"got {record.sequence}, expected {self._next_sequence}."
            )
        if not record.operation_id:
            raise CanneryProtocolError(
                "Cannery machine record has an empty operationId."
            )
        if self._operation_id is None:
            self._operation_id = record.operation_id
        elif record.operation_id != self._operation_id:
            raise CanneryProtocolError(
                "Cannery machine record operationId changed during the operation."
            )
        if record.operation not in _KNOWN_OPERATIONS:
            raise CanneryProtocolError(
                "Cannery machine record has an unsupported operation enum value."
            )
        if record.operation != _OPERATION_BY_COMMAND[self._command]:
            raise CanneryProtocolError(
                "Cannery machine record operation does not match the invoked command."
            )
        if payload is None:
            raise CanneryProtocolError("Cannery machine record is missing its payload.")
        if self._record_count == 0:
            if payload != "started":
                raise CanneryProtocolError(
                    "Cannery machine stream must begin with a started record."
                )
            self._validate_started(record.started)
        elif payload == "started":
            raise CanneryProtocolError(
                "Cannery emitted more than one started machine record."
            )

        self._next_sequence += 1
        self._record_count += 1
        if payload == "progress":
            self._observe_progress(record.progress)
        elif payload == "status":
            self._render_status("status", record.status.reason, record.status.message)
        elif payload == "warning":
            self._render_status(
                "warning", record.warning.reason, record.warning.message
            )
        elif payload == "result":
            self._terminal_kind = "result"
            self._terminal_result = self._convert_result(record.result)
        elif payload == "error":
            self._terminal_kind = "error"
            self._terminal_error = self._convert_error(record.error)
        elif payload == "cancelled":
            self._terminal_kind = "cancelled"

    def _validate_started(self, started: protocol_v1.StartedV1) -> None:
        if not started.HasField("request"):
            raise CanneryProtocolError(
                "Cannery started record is missing its command request."
            )
        request = started.request
        command = request.WhichOneof("command")
        if command is None:
            raise CanneryProtocolError(
                "Cannery started request is missing its command variant."
            )
        if (
            request.protocol_version != _PROTOCOL_VERSION
            or request.operation_id != self._operation_id
            or command != _REQUEST_BY_COMMAND[self._command]
        ):
            raise CanneryProtocolError(
                "Cannery started request metadata does not match the invocation."
            )
        if not started.cannery_version:
            raise CanneryProtocolError(
                "Cannery started record omitted the binary version."
            )

    def _observe_progress(self, progress: protocol_v1.ProgressV1) -> None:
        if progress.phase not in _PHASES_BY_COMMAND[self._command]:
            raise CanneryProtocolError(
                f"Cannery emitted an unsupported {self._command} progress phase."
            )
        if not math.isfinite(progress.elapsed_seconds) or progress.elapsed_seconds < 0:
            raise CanneryProtocolError(
                "Cannery progress elapsedSeconds must be finite and nonnegative."
            )
        self._last_phase = progress.phase
        counters = []
        for name, noun in (("files", "files"), ("bytes", "bytes"), ("items", "items")):
            done = getattr(progress, f"{name}_done")
            previous = self._progress_counters.get(name)
            if previous is not None and done < previous:
                raise CanneryProtocolError(f"Cannery progress {name}Done decreased.")
            self._progress_counters[name] = done
            total_field = f"{name}_total"
            if progress.HasField(total_field):
                total = getattr(progress, total_field)
                if done > total:
                    raise CanneryProtocolError(
                        f"Cannery progress {name}Done exceeds {name}Total."
                    )
                counters.append(f"{done}/{total} {noun}")
            elif done:
                counters.append(f"{done} {noun}")
        label = f"Cannery {_DISPLAY_OPERATION[self._command]} ({progress.phase})"
        if counters:
            label += f": {', '.join(counters)}"
        self._render_progress(label)

    def _render_status(self, kind: str, reason: str, message: str) -> None:
        if not _STABLE_REASON.fullmatch(reason):
            raise CanneryProtocolError(f"Cannery {kind} record has an invalid reason.")
        label = f"Cannery {_DISPLAY_OPERATION[self._command]}"
        if kind == "warning":
            label += " warning"
        label += f" {reason}"
        if message:
            label += f": {redact_text(message)}"
        self._render_progress(label)

    def _convert_result(self, result: protocol_v1.ResultV1) -> Dict[str, Any]:
        variant = result.WhichOneof("result")
        if variant is None:
            raise CanneryProtocolError(
                "Cannery terminal result is missing its operation variant."
            )
        expected = _REQUEST_BY_COMMAND[self._command]
        if variant != expected:
            raise CanneryProtocolError(
                "Cannery terminal result does not match the invoked operation."
            )
        value = getattr(result, variant)
        if variant == "push":
            if not value.manifest_digest or not value.canonical_reference:
                raise CanneryProtocolError(
                    "Cannery push result omitted required content identity."
                )
        elif variant == "list":
            page = value.WhichOneof("page")
            if page is None:
                raise CanneryProtocolError("Cannery list result is missing its page.")
            value = getattr(value, page)
            entries = value.namespaces if page == "namespaces" else value.references
            if len(entries) > _MAX_PAGE_ENTRIES:
                raise CanneryProtocolError(
                    "Cannery list result exceeds the v1 page limit."
                )
            if page == "references":
                for entry in entries:
                    _require_known_enum(
                        entry.kind, _KNOWN_REFERENCE_ENTRY_KINDS, "reference kind"
                    )
        elif variant == "show":
            if not value.manifest_digest or not value.canonical_reference:
                raise CanneryProtocolError(
                    "Cannery show result omitted required content identity."
                )
            if not value.HasField("file_page"):
                raise CanneryProtocolError(
                    "Cannery show result is missing its file page."
                )
            if len(value.file_page.files) > _MAX_PAGE_ENTRIES:
                raise CanneryProtocolError(
                    "Cannery show result exceeds the v1 page limit."
                )
            for entry in value.file_page.files:
                _require_known_enum(entry.kind, _KNOWN_FILE_ENTRY_KINDS, "file kind")
        elif variant == "pull":
            if (
                not value.manifest_digest
                or not value.canonical_reference
                or not value.output_directory
                or not value.content_verified
            ):
                raise CanneryProtocolError(
                    "Cannery pull result omitted required verified content metadata."
                )
        return _message_to_mapping(value)

    def _convert_error(self, error: protocol_v1.ErrorV1) -> Mapping[str, Any]:
        category = _ERROR_CATEGORY.get(error.category)
        if category is None:
            raise CanneryProtocolError(
                "Cannery error record has an unsupported category."
            )
        if not _STABLE_REASON.fullmatch(error.reason):
            raise CanneryProtocolError("Cannery error record has an invalid reason.")
        detail = error.details.WhichOneof("details")
        expected_detail = _DETAIL_BY_CATEGORY.get(error.category)
        if expected_detail is not None and detail != expected_detail:
            raise CanneryProtocolError(
                "Cannery error detail does not match its category."
            )
        if expected_detail is None and detail is not None:
            raise CanneryProtocolError(
                "Cannery error category must not include typed details."
            )
        result: TypedMachineError = TypedMachineError(
            {
                "category": category,
                "reason": error.reason,
                "message": redact_text(error.message),
                "operation": _DISPLAY_OPERATION[self._command],
            }
        )
        if error.HasField("retryable"):
            result["retryable"] = error.retryable
        if error.HasField("retry_after_ms"):
            result["retry_after_ms"] = error.retry_after_ms
        if detail is not None:
            result["details"] = _message_to_mapping(getattr(error.details, detail))
        if self._last_phase is not None:
            result["phase"] = self._last_phase
        return result

    def finish(self, return_code: int, *, enforce_exit_status: bool = True) -> None:
        stdout_errors: List[BaseException] = []
        stdout_thread: Optional[threading.Thread] = None
        if not self._stream_complete:
            stdout_thread = threading.Thread(
                target=self._drain_remaining_stdout,
                args=(stdout_errors,),
                name="truss-cannery-v1-stdout-drain",
                daemon=True,
            )
            stdout_thread.start()

        deadline = time.monotonic() + _STREAM_DRAIN_TIMEOUT_SEC
        if stdout_thread is not None:
            stdout_thread.join(max(0.0, deadline - time.monotonic()))
        self._stderr_thread.join(max(0.0, deadline - time.monotonic()))

        if stdout_errors:
            error = stdout_errors[0]
            if isinstance(error, (CanneryProtocolError, KeyboardInterrupt)):
                raise error
            raise CanneryProtocolError(
                "Cannery stdout could not be drained after process exit."
            ) from None
        if (stdout_thread is not None and stdout_thread.is_alive()) or (
            self._stderr_thread.is_alive()
        ):
            raise CanneryProtocolError(
                "Cannery output pipes did not close after process exit."
            )
        if self._stderr.error is not None:
            raise CanneryProtocolError(
                "Cannery stderr could not be drained after process exit."
            )
        if self._terminal_kind is None:
            if self._record_count == 0 and return_code == 2:
                self._terminal_kind = "error"
                self._terminal_error = {
                    "category": "usage",
                    "reason": "INVALID_ARGUMENT",
                    "operation": _DISPLAY_OPERATION[self._command],
                    "retryable": False,
                }
                return
            elif self._record_count == 0 and return_code in {130, -2}:
                self._terminal_kind = "cancelled"
                return
            else:
                raise CanneryProtocolError(
                    "Cannery machine stream ended without a terminal record."
                )
        if not enforce_exit_status:
            return
        expected_exit_code = {"result": 0, "error": 1, "cancelled": 130}[
            self._terminal_kind
        ]
        if return_code != expected_exit_code:
            raise CanneryProtocolError(
                "Cannery exit status does not match its terminal machine record; "
                f"got {return_code}, expected {expected_exit_code}."
            )

    def _drain_remaining_stdout(self, errors: List[BaseException]) -> None:
        try:
            while True:
                record = self._record_reader.read()
                if record is None:
                    self._stream_complete = True
                    return
                self._observe(record)
        except BaseException as exc:
            errors.append(exc)


class _BoundedStderrCapture:
    def __init__(self, stderr: BinaryIO) -> None:
        self._stderr = stderr
        self._chunks: Deque[str] = deque()
        self._character_count = 0
        self._truncated = False
        self.error: Optional[BaseException] = None

    @property
    def value(self) -> str:
        value = redact_text("".join(self._chunks))
        value = value[-_MAX_STDERR_CHARACTERS:]
        if self._truncated:
            return "[earlier stderr truncated]\n" + value
        return value

    def drain(self) -> None:
        try:
            decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
            reader = getattr(self._stderr, "read", None)
            chunks = (
                iter(lambda: reader(8192), b"")
                if callable(reader)
                else iter(self._stderr)
            )
            for chunk in chunks:
                if not isinstance(chunk, bytes):
                    raise TypeError("Cannery stderr is not a binary stream")
                self._append(decoder.decode(chunk))
            self._append(decoder.decode(b"", final=True))
        except BaseException as exc:
            self.error = exc

    def _append(self, text: str) -> None:
        if not text:
            return
        self._chunks.append(text)
        self._character_count += len(text)
        storage_limit = _MAX_STDERR_CHARACTERS + _STDERR_REDACTION_CONTEXT_CHARACTERS
        while self._character_count > storage_limit:
            self._truncated = True
            overflow = self._character_count - storage_limit
            first = self._chunks[0]
            if len(first) <= overflow:
                self._chunks.popleft()
                self._character_count -= len(first)
            else:
                self._chunks[0] = first[overflow:]
                self._character_count -= overflow


def _validate_symbolic_enums(document: Mapping[str, Any]) -> None:
    operation = document.get("operation")
    if operation not in _OPERATION_WIRE_NAME.values():
        raise CanneryProtocolError(
            "Cannery machine record operation must be a supported symbolic enum."
        )
    error = document.get("error")
    if isinstance(error, Mapping):
        category = error.get("category")
        if (
            not isinstance(category, str)
            or category == "ERROR_CATEGORY_UNSPECIFIED"
            or category not in protocol_v1.ErrorCategory.keys()
        ):
            raise CanneryProtocolError(
                "Cannery error category must be a supported symbolic enum."
            )
    result = document.get("result")
    if not isinstance(result, Mapping):
        return
    list_result = result.get("list")
    if isinstance(list_result, Mapping):
        references = list_result.get("references")
        if isinstance(references, Mapping):
            for entry in references.get("references", []):
                _require_symbolic_enum(
                    entry,
                    "kind",
                    protocol_v1.ReferenceEntryKind.keys(),
                    "reference kind",
                )
    show_result = result.get("show")
    if isinstance(show_result, Mapping):
        file_page = show_result.get("filePage")
        if isinstance(file_page, Mapping):
            for entry in file_page.get("files", []):
                _require_symbolic_enum(
                    entry, "kind", protocol_v1.FileEntryKind.keys(), "file kind"
                )


def _validate_required_pull_selected_totals(document: Mapping[str, Any]) -> None:
    result = document.get("result")
    if not isinstance(result, Mapping):
        return
    pull = result.get("pull")
    if not isinstance(pull, Mapping):
        return
    missing = [
        camel_name
        for camel_name, snake_name in _REQUIRED_PULL_SELECTED_TOTALS.items()
        if camel_name not in pull and snake_name not in pull
    ]
    if missing:
        raise CanneryProtocolError(
            "Cannery pull result omitted required selected totals: "
            f"{', '.join(missing)}."
        )


def _require_symbolic_enum(
    document: Any, field: str, allowed: Iterable[str], description: str
) -> None:
    if not isinstance(document, Mapping):
        raise CanneryProtocolError(f"Cannery {description} entry must be an object.")
    value = document.get(field)
    if (
        not isinstance(value, str)
        or value.endswith("_UNSPECIFIED")
        or value not in allowed
    ):
        raise CanneryProtocolError(
            f"Cannery {description} must be a supported symbolic enum."
        )


def _require_known_enum(value: int, allowed: Iterable[int], description: str) -> None:
    if value not in allowed:
        raise CanneryProtocolError(
            f"Cannery {description} has an unsupported enum value."
        )


def _message_to_mapping(message: Message) -> Dict[str, Any]:
    return json_format.MessageToDict(
        message,
        preserving_proto_field_name=True,
        use_integers_for_enums=False,
        always_print_fields_with_no_presence=True,
    )
