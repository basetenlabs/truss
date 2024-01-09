import inspect
from types import MappingProxyType
from typing import (
    Any,
    AsyncGenerator,
    Awaitable,
    Generator,
    Optional,
    Tuple,
    Union,
    get_args,
    get_origin,
)

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

        input_type = _parse_input_type(input_parameters)
        ouput_type = _parse_output_type(output_annotation)

        if not input_type or not ouput_type:
            return None

        output_type, supports_streaming = ouput_type

        return cls(
            input_type=input_type,
            output_type=output_type,
            supports_streaming=supports_streaming,
        )


def _parse_input_type(input_parameters: MappingProxyType) -> Optional[type]:
    parameter_types = list(input_parameters.values())

    if len(parameter_types) > 1:
        return None

    input_type = parameter_types[0].annotation

    if (
        input_type == inspect.Signature.empty
        or not isinstance(input_type, type)
        or not issubclass(input_type, BaseModel)
    ):
        return None

    return input_type


def _parse_output_type(output_annotation: Any) -> Optional[Tuple[Optional[type], bool]]:
    """
    Therea are 4 possible cases for output_annotation:
    1. Data object -- represented by a Pydantic BaseModel
    2. Streaming -- represented by a Generator or AsyncGenerator
    3. Async -- represented by an Awaitable
    4. Streaming or data object -- it is possible for a function to return both
       a data object and a streaming object. This is represented by a Union of
       a Pydantic BaseModel and a Generator or AsyncGenerator

    If the output_annotation does not match one of these cases, returns None
    """
    if isinstance(output_annotation, type) and issubclass(output_annotation, BaseModel):
        return output_annotation, False

    if _is_generator_type(output_annotation):
        return None, True

    if _is_awaitable_type(output_annotation):
        output_type = retrieve_base_class_from_awaitable(output_annotation)
        if not output_type:
            return None
        return output_type, False

    if _is_union_type(output_annotation):
        output_type = retrieve_base_class_from_union(output_annotation)
        if not output_type:
            return None
        return output_type, True

    return None


def _is_generator_type(annotation: Any) -> bool:
    base_type = get_origin(annotation)
    return isinstance(base_type, type) and issubclass(
        base_type, (Generator, AsyncGenerator)
    )


def _is_union_type(annotation: Any) -> bool:
    return get_origin(annotation) == Union


def _is_awaitable_type(annotation: Any) -> bool:
    base_type = get_origin(annotation)
    return isinstance(base_type, type) and issubclass(base_type, Awaitable)


def retrieve_base_class_from_awaitable(awaitable_annotation: type) -> Optional[type]:
    """
    Returns the base class of an Awaitable type if it is of the form:
    Awaitable[PydanticBaseModel]

    Else, returns None
    """
    (awaitable_type,) = get_args(
        awaitable_annotation
    )  # Note that Awaitable has only one type argument
    if isinstance(awaitable_type, type) and issubclass(awaitable_type, BaseModel):
        return awaitable_type

    return None


def _extract_pydantic_base_models(union_args: tuple) -> list:
    """
    Extracts any pydantic base model arguments from the arms of a Union type.
    The two cases are:
    1. Union[PydanticBaseModel, Generator]
    2. Union[Awaitable[PydanticBaseModel], AsyncGenerator]
    So for Awaitables, we need to extract the base class from the Awaitable type
    """
    return [
        retrieve_base_class_from_awaitable(arg) if _is_awaitable_type(arg) else arg
        for arg in union_args
        if _is_awaitable_type(arg)
        or (isinstance(arg, type) and issubclass(arg, BaseModel))
    ]


def retrieve_base_class_from_union(union_annotation: type) -> Optional[type]:
    """
    Returns the base class of a Union type if it is of the form:
    Union[PydanticBaseModel, Generator] or in the case of async functions:
    Union[Awaitable[PydanticBaseModel], AsyncGenerator]

    Else, returns None
    """
    union_args = get_args(union_annotation)

    if len(union_args) != 2:
        return None

    pydantic_base_models = _extract_pydantic_base_models(union_args)
    generators = [arg for arg in union_args if _is_generator_type(arg)]

    if len(pydantic_base_models) != 1 or len(generators) != 1:
        return None

    return pydantic_base_models[0]
