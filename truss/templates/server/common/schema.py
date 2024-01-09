import inspect
from types import MappingProxyType
from typing import Any, AsyncGenerator, Awaitable, Generator, Optional, Union

from pydantic import BaseModel


class TrussSchema(BaseModel):
    input_type: type
    output_type: Optional[type]
    supports_streaming: bool

    @classmethod
    def from_signature(
        cls, input_parameters: MappingProxyType, output_annotation: Any
    ) -> Optional["TrussSchema"]:
        """
        Create a TrussSchema from a function signature if annotated, else returns None
        """
        parameter_types = list(input_parameters.values())

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

        if isinstance(output_annotation, type) and issubclass(
            output_annotation, BaseModel
        ):
            output_type = output_annotation
            supports_streaming = False
        elif _is_union_type(output_annotation):
            # Check both types in the union are valid:
            output_type = retrieve_base_class_from_union(output_annotation)
            if not output_type:
                return None
            supports_streaming = True
        elif _is_generator_type(output_annotation):
            output_type = None

            supports_streaming = True
        elif _is_awaitable_type(output_annotation):
            output_type = retrieve_base_class_from_awaitable(output_annotation)
            if not output_type:
                return None
            supports_streaming = False

        else:
            return None

        return cls(
            input_type=input_type,
            output_type=output_type,
            supports_streaming=supports_streaming,
        )


def _is_generator_type(annotation: type) -> bool:
    return hasattr(annotation, "__origin__") and issubclass(
        annotation.__origin__, (Generator, AsyncGenerator)
    )


def _is_union_type(annotation: type) -> bool:
    return hasattr(annotation, "__origin__") and annotation.__origin__ == Union


def _is_awaitable_type(annotation: type) -> bool:
    return hasattr(annotation, "__origin__") and issubclass(
        annotation.__origin__, Awaitable
    )


def retrieve_base_class_from_awaitable(awaitable_annotation: type) -> Optional[type]:
    """
    Returns the base class of an Awaitable type if it is of the form:
    Awaitable[PydanticBaseModel]

    Else, returns None
    """
    (awaitable_type,) = awaitable_annotation.__args__  # type: ignore
    if isinstance(awaitable_type, type) and issubclass(awaitable_type, BaseModel):
        return awaitable_type

    return None


def _extract_pydantic_base_models(union_args: tuple) -> list:
    return [
        retrieve_base_class_from_awaitable(arg) if _is_awaitable_type(arg) else arg
        for arg in union_args
        if (_is_awaitable_type(arg) and retrieve_base_class_from_awaitable(arg))
        or (isinstance(arg, type) and issubclass(arg, BaseModel))
    ]


def retrieve_base_class_from_union(union_annotation: type) -> Optional[type]:
    """
    Returns the base class of a Union type if it is of the form:
    Union[PydanticBaseModel, Generator] or in the case of async functions:
    Union[Awaitable[PydanticBaseModel], AsyncGenerator]

    Else, returns None
    """
    union_args = union_annotation.__args__  # type: ignore

    if len(union_args) != 2:
        return None

    pydantic_base_models = _extract_pydantic_base_models(union_args)
    generators = [arg for arg in union_args if _is_generator_type(arg)]

    if len(pydantic_base_models) != 1 or len(generators) != 1:
        return None

    return pydantic_base_models[0]
