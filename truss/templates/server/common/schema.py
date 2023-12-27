import inspect
from typing import AsyncGenerator, Generator, Optional, Union

from pydantic import BaseModel


class TrussSchema(BaseModel):
    input_type: type
    output_type: type
    supports_streaming: bool

    @classmethod
    def from_signature(cls, signature: inspect.Signature) -> Optional["TrussSchema"]:
        """
        Create a TrussSchema from a function signature if annotated, else returns None
        """
        parameter_types = list(signature.parameters.values())

        if len(parameter_types) > 1:
            return None

        input_type = parameter_types[0].annotation
        output_type = None
        supports_streaming = False

        if (
            input_type == inspect.Signature.empty
            or not isinstance(input_type, type)
            or not issubclass(input_type, BaseModel)
        ):
            return None

        if issubclass(signature.return_annotation, BaseModel):
            output_type = signature.return_annotation
            supports_streaming = False
        elif signature.return_annotation == Generator:
            output_type = None
            supports_streaming = True
        elif _is_union_type(signature.return_annotation):
            # Check both types in the union are valid:
            output_type = retrieve_base_class_from_union(signature.return_annotation)
            supports_streaming = True
        else:
            return None

        return cls(
            input_type=input_type,
            output_type=output_type,
            supports_streaming=supports_streaming,
        )


def _is_union_type(annotation: type) -> bool:
    return hasattr(annotation, "__origin__") and annotation.__origin__ == Union


def retrieve_base_class_from_union(union_annotation: type) -> Optional[type]:
    """
    Returns the base class of a Union type if it is of the form:
    Union[PydanticBaseModel, Generator]

    Else, returns None
    """
    union_args = union_annotation.__args__  # type: ignore

    if len(union_args) != 2:
        return None

    pydantic_base_models = [
        arg
        for arg in union_args
        if isinstance(arg, type) and issubclass(arg, BaseModel)
    ]
    generators = [
        arg
        for arg in union_args
        if issubclass(arg, Generator) or issubclass(arg, AsyncGenerator)
    ]

    if len(pydantic_base_models) != 1 or len(generators) != 1:
        return None

    return pydantic_base_models[0]
