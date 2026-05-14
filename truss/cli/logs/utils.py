import re
from datetime import datetime, timedelta, timezone
from typing import Any, List, Optional, Tuple

import pydantic
import rich_click as click
from rich import text

from truss.cli.utils.output import console

MAX_LOG_RANGE = timedelta(days=7)

_SINCE_RE = re.compile(r"^(\d+)([mhd])$")
_SINCE_UNIT_TO_SECONDS = {"m": 60, "h": 3600, "d": 86400}


class ParsedLog(pydantic.BaseModel):
    timestamp: str
    message: str
    replica: Optional[str]

    @classmethod
    def from_raw(self, raw_log: Any) -> "ParsedLog":
        return ParsedLog(**raw_log)


def parse_since(value: str) -> timedelta:
    """Parse a `--since` duration like '30m', '2h', '3d' into a timedelta.

    Accepts integer values with a single unit suffix: m (minutes), h (hours),
    or d (days). Rejects zero, negative, malformed, or out-of-range values.
    Maximum allowed duration is MAX_LOG_RANGE (7 days).
    """
    match = _SINCE_RE.match(value.strip()) if value else None
    if not match:
        raise click.BadParameter(
            f"Invalid --since value '{value}'. Expected format <N><unit> where "
            f"unit is m (minutes), h (hours), or d (days). Examples: '30m', '2h', '3d'."
        )

    amount = int(match.group(1))
    if amount <= 0:
        raise click.BadParameter("--since must be greater than zero.")

    delta = timedelta(seconds=amount * _SINCE_UNIT_TO_SECONDS[match.group(2)])
    if delta > MAX_LOG_RANGE:
        raise click.BadParameter(f"--since must be at most 7d (got '{value}').")
    return delta


def _to_epoch_millis(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def resolve_log_time_range(
    start: Optional[datetime], end: Optional[datetime], since: Optional[str]
) -> Tuple[Optional[int], Optional[int]]:
    """Resolve --start/--end/--since into (start_epoch_millis, end_epoch_millis).

    Returns (None, None) when none of the inputs are supplied. Naive datetimes
    are interpreted as UTC. Enforces a 7-day maximum window and rejects
    combining --since with --start/--end.
    """
    if since is not None and (start is not None or end is not None):
        raise click.UsageError("--since cannot be combined with --start or --end.")

    if since is not None:
        delta = parse_since(since)
        now = datetime.now(timezone.utc)
        return _to_epoch_millis(now - delta), _to_epoch_millis(now)

    if start is None and end is None:
        return None, None

    if end is None:
        end = datetime.now(timezone.utc)
    if start is None:
        start = end - MAX_LOG_RANGE

    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)

    if start >= end:
        raise click.UsageError("--start must be earlier than --end.")
    if end - start > MAX_LOG_RANGE:
        raise click.UsageError(
            "Log time range must be at most 7 days. Narrow --start/--end or use --since."
        )

    return _to_epoch_millis(start), _to_epoch_millis(end)


def format_log(log: ParsedLog) -> str:
    """Format a log entry as a string."""
    epoch_nanos = int(log.timestamp)
    dt = datetime.fromtimestamp(epoch_nanos / 1e9)
    formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S")

    replica_part = f"({log.replica}) " if log.replica else ""
    return f"[{formatted_time}]: {replica_part}{log.message.strip()}"


def output_log(log: ParsedLog):
    epoch_nanos = int(log.timestamp)
    dt = datetime.fromtimestamp(epoch_nanos / 1e9)
    formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S")

    time_text = text.Text(f"[{formatted_time}]: ", style="blue")
    message_text = text.Text(f"{log.message.strip()}")
    if log.replica:
        replica_text = text.Text(f"({log.replica}) ", style="green")
    else:
        replica_text = text.Text()

    # Output the combined log line to the console
    console.print(time_text, replica_text, message_text, sep="")


def parse_logs(api_logs: List[Any]) -> List[ParsedLog]:
    return [ParsedLog.from_raw(api_log) for api_log in api_logs]
