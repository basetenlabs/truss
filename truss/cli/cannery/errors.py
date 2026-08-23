from __future__ import annotations

import enum
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Union

import rich_click as click

from truss.cli.cannery.diagnostics import diagnostic_failure_suffix


class CanneryClickException(click.ClickException):
    """An error whose message contains only reviewed wrapper-generated text."""


class CanneryUsageError(click.UsageError):
    """A usage error whose message contains only reviewed wrapper-generated text."""


class CanneryProtocolError(CanneryClickException):
    """The Cannery subprocess violated its selected machine protocol."""


class ErrorCategory(str, enum.Enum):
    USAGE = "usage"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    NOT_FOUND = "not_found"
    NETWORK = "network"
    THROTTLED = "throttled"
    QUOTA = "quota"
    SERVER = "server"
    INTEGRITY = "integrity"
    LOCAL_IO = "local_io"
    INCOMPATIBLE_CLIENT = "incompatible_client"


@dataclass(frozen=True)
class RetryInfo:
    delay_sec: float


_REASON_CATEGORY = {
    "invalid_ref": ErrorCategory.USAGE,
    "invalid_argument": ErrorCategory.USAGE,
    "unauthenticated": ErrorCategory.AUTHENTICATION,
    "unauthorized": ErrorCategory.AUTHENTICATION,
    "permission_denied": ErrorCategory.AUTHORIZATION,
    "forbidden": ErrorCategory.AUTHORIZATION,
    "not_found": ErrorCategory.NOT_FOUND,
    "rate_limited": ErrorCategory.THROTTLED,
    "resource_exhausted": ErrorCategory.QUOTA,
}
_MACHINE_IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")


def error_category(
    error: Optional[Mapping[str, Any]], return_code: int
) -> ErrorCategory:
    if error is None:
        return ErrorCategory.SERVER
    raw_category = error.get("category")
    if isinstance(raw_category, str):
        try:
            return ErrorCategory(raw_category.lower())
        except ValueError:
            return ErrorCategory.SERVER
    reason = error.get("reason") or error.get("status")
    if isinstance(reason, str):
        inferred = _REASON_CATEGORY.get(reason.lower())
        if inferred is not None:
            return inferred
    if return_code == 2:
        return ErrorCategory.USAGE
    return ErrorCategory.SERVER


def retry_info(error: Optional[Mapping[str, Any]]) -> Optional[RetryInfo]:
    if error is None:
        return None
    details = error.get("retry_info") or error.get("retryInfo")
    if not isinstance(details, Mapping):
        details = error
    for key in (
        "retry_after_seconds",
        "retryAfterSeconds",
        "delay_seconds",
        "delaySeconds",
    ):
        value = details.get(key)
        parsed = _nonnegative_float(value)
        if parsed is not None:
            return RetryInfo(parsed)
    delay = details.get("retry_delay") or details.get("retryDelay")
    if isinstance(delay, Mapping):
        seconds = _nonnegative_float(delay.get("seconds")) or 0.0
        nanos = _nonnegative_float(delay.get("nanos")) or 0.0
        return RetryInfo(seconds + nanos / 1_000_000_000)
    if isinstance(delay, str) and delay.endswith("s"):
        parsed = _nonnegative_float(delay[:-1])
        if parsed is not None:
            return RetryInfo(parsed)
    return None


def _nonnegative_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def command_failure(
    error: Optional[Mapping[str, Any]],
    return_code: int,
    correlation_id: str,
    diagnostic_path: Optional[Path],
) -> click.ClickException:
    category = error_category(error, return_code)
    rendered = _format_machine_error(error, return_code)
    rendered += f" Category: {category.value}."
    if error is not None:
        operation = safe_machine_identifier(error.get("operation"))
        phase = safe_machine_identifier(error.get("phase"))
        if operation is not None:
            rendered += f" Operation: {operation}."
        if phase is not None:
            rendered += f" Phase: {phase}."
        retryable = error.get("retryable")
        if isinstance(retryable, bool):
            rendered += f" Retryable: {'yes' if retryable else 'no'}."

    retry = retry_info(error)
    if category == ErrorCategory.THROTTLED and retry is not None:
        rendered += f" Retry after {retry.delay_sec:g} seconds."
    elif category == ErrorCategory.QUOTA:
        rendered += " Retrying will not help until quota is available."

    rendered += " " + diagnostic_failure_suffix(correlation_id, diagnostic_path)
    if category == ErrorCategory.USAGE or return_code == 2:
        return CanneryUsageError(rendered)
    return CanneryClickException(rendered)


def attach_failure_context(
    error: Union[CanneryClickException, CanneryUsageError],
    correlation_id: str,
    diagnostic_path: Optional[Path],
) -> click.ClickException:
    suffix = diagnostic_failure_suffix(correlation_id, diagnostic_path)
    if suffix not in error.message:
        error.message = f"{error.message} {suffix}"
    return error


def _format_machine_error(error: Optional[Mapping[str, Any]], return_code: int) -> str:
    if error is None:
        return (
            f"Cannery exited with status {return_code} without a machine error event."
        )

    reason = safe_machine_identifier(error.get("reason") or error.get("status"))
    if reason is not None:
        return f"Cannery failed with reason {reason}"
    return f"Cannery exited with status {return_code}"


def safe_machine_identifier(value: Any) -> Optional[str]:
    if isinstance(value, str) and _MACHINE_IDENTIFIER.fullmatch(value):
        return value
    return None
