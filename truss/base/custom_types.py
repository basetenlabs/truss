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
    # In general, we cannot be strict with forbidding extra keys, because server-side
    # we might build the context with an older version that doesn't know these keys.
    # Therefore we only check extra keys in a special method `validate_forbid_extra`
    # that is invoked during push (CLI side).
    # But in order to detect the keys then, we need to `allow` them at initial parsing,
    # this way they will be included in the dump, but not on the programmatic API.
    model_config = pydantic.ConfigDict(
        validate_assignment=True, extra="allow", arbitrary_types_allowed=False
    )

    @pydantic.model_validator(mode="before")
    def _maybe_forbid_extra(cls, data: dict, info: pydantic.ValidationInfo) -> dict:
        if info.context and info.context.get("forbid_extra"):
            extra_fields = set(data) - set(cls.model_fields)
            if extra_fields:
                raise ValueError(
                    f"Extra fields not allowed: [{', '.join(sorted(extra_fields))}]. Possibly "
                    "this field is only supported by newer truss versions, check for updates."
                )
        return data

    def validate_forbid_extra(self) -> None:
        type(self).model_validate(
            self.model_dump(mode="json"), context={"forbid_extra": True}
        )

    def to_dict(self, verbose: bool = False) -> dict[str, Any]:
        kwargs: dict[str, Any] = (
            {"mode": "json"}
            if verbose
            else {
                "mode": "json",
                "exclude_unset": False,
                "exclude_none": True,
                "exclude_defaults": True,
                "context": {"verbose": verbose},
            }
        )
        return super().model_dump(**kwargs)
