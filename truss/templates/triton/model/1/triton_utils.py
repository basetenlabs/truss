import inspect
import json
from typing import List

import numpy as np
import triton_python_backend_utils as pb_utils

TYPE_CONVERSION_LAMBDAS = {
    "dict": lambda x: json.loads(x.as_numpy().item().decode("utf-8")),
    "str": lambda x: x.as_numpy().item().decode("utf-8"),
    "int": lambda x: x.as_numpy().item(),
    "float": lambda x: x.as_numpy().item(),
    "bool": lambda x: x.as_numpy().item(),
    "list": lambda x: x,
}

PYTHON_TO_NP_DTYPES = {
    int: np.int32,
    float: np.float32,
    str: np.dtype(object),
    bool: np.bool_,
    dict: np.dtype(object),
}


def _signature_accepts_keyword_arg(signature: inspect.Signature, kwarg: str) -> bool:
    """Checks if a signature accepts a keyword argument with the given name."""
    return kwarg in signature.parameters or _signature_accepts_kwargs(signature)


def _signature_accepts_kwargs(signature: inspect.Signature) -> bool:
    """Checks if a signature accepts **kwargs."""
    for param in signature.parameters.values():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            return True
    return False


def _is_list_type(pydantic_field_annotation) -> bool:
    """
    Checks if the provided type is List type.
    """
    actual_type = getattr(
        pydantic_field_annotation, "outer_type_", pydantic_field_annotation
    )
    return actual_type == List or (
        hasattr(actual_type, "__origin__") and actual_type.__origin__ == list
    )


def _get_pydantic_type(pydantic_field):
    if _is_list_type(pydantic_field):
        return list
    else:
        return (
            pydantic_field.__name__
            if hasattr(pydantic_field, "__name__")
            else pydantic_field.__origin__.__name__
        )


def _inspect_pydantic_model(pydantic_class):
    """
    Inspects a Pydantic model and returns a list of fields and their types.
    """
    if not pydantic_class:
        raise ValueError("Model not instantiated or input_type not set")

    return [
        {"param": field_name, "type": _get_pydantic_type(field_info.annotation)}
        for field_name, field_info in pydantic_class.model_fields.items()
    ]


def _transform_triton_requests_to_pydantic_type(triton_objects, pydantic_type):
    """
    Transforms a list of Triton InferenceRequests to a list of Pydantic objects.
    """
    pydantic_type_fields = _inspect_pydantic_model(pydantic_type)

    coerced_requests = []
    for triton_object in triton_objects:
        coerced_request = {
            field["param"]: TYPE_CONVERSION_LAMBDAS[field["type"]](
                pb_utils.get_input_tensor_by_name(triton_object, field["param"])
            )
            for field in pydantic_type_fields
        }
        coerced_requests.append(pydantic_type(**coerced_request))

    return coerced_requests


def _transform_pydantic_type_to_triton(pydantic_objects):
    """
    Transforms a list of Pydantic objects to a list of Triton requests.
    """
    triton_objects = []

    for pydantic_object in pydantic_objects:
        triton_object = []
        for field_name, field_value in pydantic_object.dict().items():
            if isinstance(field_value, list):
                # Ensure all elements are of the same type using the type of the first element
                if not all(isinstance(x, (int, float, str, bool)) for x in field_value):
                    raise ValueError(
                        "All elements of the list must be of the same type (int, float, str, bool)."
                    )
                np_dtype = PYTHON_TO_NP_DTYPES[type(field_value[0])]
                field_value = np.array(field_value, dtype=np_dtype)
            elif isinstance(field_value, dict):
                np_dtype = PYTHON_TO_NP_DTYPES.get(type(field_value))
                field_value = json.dumps(field_value)
                field_value = np.array([field_value], dtype=np_dtype)
            else:
                np_dtype = PYTHON_TO_NP_DTYPES.get(type(field_value))
                if not np_dtype:
                    raise TypeError(
                        f"Unsupported Python data type for numpy conversion: {field_value}"
                    )
                field_value = np.array([field_value], dtype=np_dtype)

            triton_object.append(pb_utils.Tensor(field_name, field_value))

        triton_objects.append(pb_utils.InferenceResponse(triton_object))

    return triton_objects
