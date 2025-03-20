from typing import Any, Callable, ContextManager, Optional, Protocol

import pydantic


# NB(nikhil): Helper function to output text with a spinner, intended
# to remove dependency on `rich` outside the `cli` packages.
class Spinner(Protocol):
    def update(self, text: str) -> None: ...


SpinnerFactory = Callable[[str], ContextManager[Spinner]]


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

    @classmethod
    def from_raw(self, raw_log: Any) -> "ParsedLog":
        return ParsedLog(**raw_log)
