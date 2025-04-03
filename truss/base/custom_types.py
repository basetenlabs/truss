from dataclasses import dataclass
from typing import Any

import pydantic


@dataclass
class Example:
    name: str
    input: Any

    @staticmethod
    def from_dict(example_dict):
        return Example(name=example_dict["name"], input=example_dict["input"])

    def to_dict(self) -> dict:
        return {"name": self.name, "input": self.input}


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


class ConfigModel(pydantic.BaseModel):
    def to_dict(self, verbose: bool = False) -> dict[str, Any]:
        kwargs: dict[str, Any] = (
            {"mode": "json"}
            if verbose
            else {
                "mode": "json",
                "exclude_unset": True,
                "exclude_none": True,
                "exclude_defaults": True,
                "context": {"verbose": verbose},
            }
        )
        return super().model_dump(**kwargs)
