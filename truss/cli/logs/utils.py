from datetime import datetime
from typing import Any, List, Optional

import pydantic
from rich import text

from truss.cli.utils.output import console


class ParsedLog(pydantic.BaseModel):
    timestamp: str
    message: str
    replica: Optional[str]

    @classmethod
    def from_raw(self, raw_log: Any) -> "ParsedLog":
        return ParsedLog(**raw_log)


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
