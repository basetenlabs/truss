from datetime import datetime
from typing import Any, List, Optional

import pydantic
import rich
from rich import text


class RawTrainingJobLog(pydantic.BaseModel):
    timestamp: str
    message: str
    replica: Optional[str]


def _output_log(log: RawTrainingJobLog, console: "rich.Console"):
    epoch_nanos = int(log.timestamp)
    dt = datetime.fromtimestamp(epoch_nanos / 1e9)
    formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S")

    time_text = text.Text(f"[{formatted_time}]", style="blue")

    assert log.replica is not None
    replica_text = text.Text(f" ({log.replica})", style="green")
    message_text = text.Text(f": {log.message.strip()}", style="white")

    # Output the combined log line to the console
    console.print(time_text, replica_text, message_text, sep="")


def format_and_output_logs(api_logs: List[Any], console: "rich.Console"):
    logs = [RawTrainingJobLog(**log) for log in api_logs]
    # The API returns the most recent results first, but users expect
    # to see those at the bottom.
    for log in logs[::-1]:
        _output_log(log, console)
