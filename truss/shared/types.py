from datetime import datetime
from typing import Any, Optional

import pydantic
import rich
from rich import text


class SafeModel(pydantic.BaseModel):
    """Pydantic base model with reasonable config."""

    model_config = pydantic.ConfigDict(
        arbitrary_types_allowed=False, strict=True, validate_assignment=True
    )


class SafeModelNonSerializable(pydantic.BaseModel):
    """Pydantic base model with reasonable config - allowing arbitrary types."""

    model_config = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
        strict=True,
        validate_assignment=True,
        extra="forbid",
    )


class ParsedLog(pydantic.BaseModel):
    timestamp: str
    message: str
    replica: Optional[str]

    def to_console(self, console: "rich.Console") -> None:
        epoch_nanos = int(self.timestamp)
        dt = datetime.fromtimestamp(epoch_nanos / 1e9)
        formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S")

        time_text = text.Text(f"[{formatted_time}]: ", style="blue")
        message_text = text.Text(f"{self.message.strip()}", style="white")
        if self.replica:
            replica_text = text.Text(f"({self.replica}) ", style="green")
        else:
            replica_text = text.Text()

        # Output the combined log line to the console
        console.print(time_text, replica_text, message_text, sep="")

    @classmethod
    def from_raw(self, raw_log: Any) -> "ParsedLog":
        return ParsedLog(**raw_log)
