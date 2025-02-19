import importlib
from typing import Type

import pydantic


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


def get_spec_from_file(config: str, cls: Type) -> Type:
    """Get a spec from a file."""
    spec = importlib.util.spec_from_file_location("config_module", config)
    if not spec or not spec.loader:
        raise ValueError(f"Could not load {config}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name, obj in module.__dict__.items():
        if isinstance(obj, cls):
            return obj
    raise ValueError(f"Could not find {cls} in {config}")
