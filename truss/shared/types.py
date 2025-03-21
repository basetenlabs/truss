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
