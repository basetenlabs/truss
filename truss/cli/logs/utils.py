from datetime import datetime
from typing import Any, List, NamedTuple, Optional

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


class LogComponents(NamedTuple):
    time_text: str
    replica_text: str
    message_text: str


def format_log(log: ParsedLog) -> LogComponents:
    epoch_nanos = int(log.timestamp)
    dt = datetime.fromtimestamp(epoch_nanos / 1e9)
    formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S")

    time_text = f"[{formatted_time}]: "
    message_text = f"{log.message.strip()}"
    if log.replica:
        replica_text = f"({log.replica}) "
    else:
        replica_text = ""

    return LogComponents(time_text, replica_text, message_text)


def output_log(log: ParsedLog):
    time_text, replica_text, message_text = format_log(log)

    styled_time = text.Text(time_text, style="blue")
    styled_replica = (
        text.Text(replica_text, style="green") if replica_text else text.Text()
    )
    styled_message = text.Text(message_text)

    console.print(styled_time, styled_replica, styled_message, sep="")


def parse_logs(api_logs: List[Any]) -> List[ParsedLog]:
    return [ParsedLog.from_raw(api_log) for api_log in api_logs]
