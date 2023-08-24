from typing import Any, List, Type, Union

import numpy as np
from pydantic import BaseModel
from utils.errors import UnsupportedTypeError

# This is a broad type that includes built-in Python types and types that might be
# returned by Pydantic functions like conlist.
FieldAnnotationType = Union[
    Type[int], Type[float], Type[str], Type[bool], Type[dict], List[Any]
]

# Dictionary to specify the numpy dtype when converting from Python types to Triton tensors
PYTHON_TYPE_TO_NP_DTYPE = {
    int: np.int32,
    float: np.float32,
    str: np.dtype(object),
    bool: np.bool_,
    dict: np.dtype(object),
}


def is_conlist(pydantic_field_annotation: FieldAnnotationType) -> bool:
    """Checks if a Pydantic field annotation is a conlist."""
    # Annotations that are not directly a Python type have an __origin__ attribute
    if not hasattr(pydantic_field_annotation, "__origin__"):
        return False
    return pydantic_field_annotation.__origin__ == list


def get_pydantic_field_type(pydantic_field_annotation: FieldAnnotationType) -> str:
    """
    Returns the type of a Pydantic field via it's annotation. The annotation can be a
    type or a type with constraints (e.g. conlist). For the following types on a Pydantic
    field, the pydantic_field_annotation is

    - str: str
    - int: int
    - float: float
    - bool: bool
    - list: list
    - conlist: typing.List[...] where ... is the type of the list elements

    Args:
        pydantic_field (FieldAnnotationType): The Pydantic field.

    Returns:
        The type (str) of the Pydantic field.
    """
    if isinstance(pydantic_field_annotation, type):
        return pydantic_field_annotation.__name__
    elif is_conlist(pydantic_field_annotation):
        return "list"
    else:
        raise UnsupportedTypeError(
            f"Unsupported Pydantic field annotation type: {pydantic_field_annotation}"
        )


def inspect_pydantic_model(pydantic_class: Type[BaseModel]) -> list:
    """
    Inspects a Pydantic model and returns a list of fields and their types.

    Args:
        pydantic_class (Type[BaseModel]): The Pydantic model class.

    Returns:
        A list of dicts containing the field name and type.
    """
    return [
        {"param": field_name, "type": get_pydantic_field_type(field_info.annotation)}
        for field_name, field_info in pydantic_class.__fields__.items()  # type: ignore
    ]
