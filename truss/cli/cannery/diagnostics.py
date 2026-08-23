from __future__ import annotations

import json
import os
import re
import stat
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional
from urllib.parse import urlsplit

import rich_click as click

_SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b(authorization|api[_-]?key|access[_-]?token|refresh[_-]?token|"
    r"bearer[_-]?token|secret)\b(\s*[:=]\s*)([^\s,;]+)"
)
_BEARER_VALUE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/\-]+=*")
_SAFE_FIELDS = frozenset(
    {
        "artifact_sha256",
        "artifact_version",
        "binary_path",
        "category",
        "duration_sec",
        "endpoint_hostname",
        "exception_class",
        "exit_code",
        "mechanism",
        "message",
        "operating_system",
        "operation",
        "phase",
        "protocol_version",
        "reason",
        "retryable",
    }
)


def redact_text(value: str) -> str:
    redacted = _BEARER_VALUE.sub("Bearer [REDACTED]", value)
    return _SECRET_ASSIGNMENT.sub(
        lambda match: f"{match.group(1)}{match.group(2)}[REDACTED]", redacted
    )


def endpoint_hostname(endpoint: str) -> str:
    parsed = urlsplit(endpoint)
    return parsed.hostname or "<invalid>"


def _sanitize_value(value: Any) -> Any:
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _sanitize_value(item)
            for key, item in value.items()
            if str(key) in _SAFE_FIELDS
        }
    if isinstance(value, (list, tuple)):
        return [_sanitize_value(item) for item in value]
    return redact_text(str(value))


class DiagnosticLog:
    def __init__(self, path: Path, correlation_id: str) -> None:
        self.path = path
        self.correlation_id = correlation_id

    @classmethod
    def create(cls, correlation_id: str) -> "DiagnosticLog":
        directory = _diagnostic_directory()
        _prepare_private_directory(directory)
        descriptor, name = tempfile.mkstemp(
            prefix="diagnostic-", suffix=".jsonl", dir=directory
        )
        try:
            if hasattr(os, "fchmod"):
                os.fchmod(descriptor, 0o600)
        finally:
            os.close(descriptor)
        return cls(Path(name), correlation_id)

    def record(self, event: str, **fields: Any) -> None:
        payload: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "correlation_id": self.correlation_id,
        }
        payload.update(
            {
                key: _sanitize_value(value)
                for key, value in fields.items()
                if key in _SAFE_FIELDS
            }
        )
        path_stat = _validate_private_log(self.path)
        flags = os.O_WRONLY | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(self.path, flags)
        opened_stat = os.fstat(descriptor)
        if (opened_stat.st_dev, opened_stat.st_ino) != (
            path_stat.st_dev,
            path_stat.st_ino,
        ):
            os.close(descriptor)
            raise click.ClickException(
                "Cannery diagnostic log changed while it was being written."
            )
        with os.fdopen(descriptor, "a", encoding="utf-8") as diagnostic_file:
            diagnostic_file.write(json.dumps(payload, sort_keys=True))
            diagnostic_file.write("\n")
            diagnostic_file.flush()
            os.fsync(diagnostic_file.fileno())

    def delete(self) -> None:
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


def _diagnostic_directory() -> Path:
    configured = os.environ.get("TRUSS_CANNERY_DIAGNOSTIC_DIR")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".cache" / "truss" / "cannery" / "diagnostics"


def _prepare_private_directory(directory: Path) -> None:
    try:
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    except OSError:
        raise click.ClickException(
            "Could not create the private Cannery diagnostics directory."
        ) from None
    directory_stat = directory.lstat()
    if stat.S_ISLNK(directory_stat.st_mode) or not stat.S_ISDIR(directory_stat.st_mode):
        raise click.ClickException(
            "Cannery diagnostics path must be a regular directory, not a symlink."
        )
    if os.name != "nt" and directory_stat.st_uid != os.getuid():
        raise click.ClickException(
            "Cannery diagnostics directory must be owned by the current user."
        )
    if os.name != "nt" and directory_stat.st_mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise click.ClickException(
            "Cannery diagnostics directory must be owner-only (mode 0700)."
        )


def _validate_private_log(path: Path) -> os.stat_result:
    path_stat = path.lstat()
    if stat.S_ISLNK(path_stat.st_mode) or not stat.S_ISREG(path_stat.st_mode):
        raise click.ClickException(
            "Cannery diagnostic log must be a regular file, not a symlink."
        )
    if os.name != "nt" and path_stat.st_uid != os.getuid():
        raise click.ClickException(
            "Cannery diagnostic log must be owned by the current user."
        )
    if os.name != "nt" and path_stat.st_mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise click.ClickException(
            "Cannery diagnostic log must be owner-only (mode 0600)."
        )
    return path_stat


def diagnostic_failure_suffix(
    correlation_id: str, diagnostic_path: Optional[Path]
) -> str:
    rendered = f"Correlation ID: {correlation_id}."
    if diagnostic_path is not None:
        rendered += f" Diagnostics: {diagnostic_path}."
    return rendered
