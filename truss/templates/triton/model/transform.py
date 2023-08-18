import json
from typing import List, Type

import numpy as np
import triton_python_backend_utils as pb_utils
from pydantic import BaseModel
from utils.pydantic import PYTHON_TYPE_TO_NP_DTYPE, inspect_pydantic_model
from utils.triton import (
    convert_tensor_to_python_type,
    create_inference_response,
    create_tensor,
    get_input_tensor_by_name,
)


def transform_triton_to_pydantic(
    triton_requests: List["pb_utils.InferenceRequest"],
    pydantic_type: Type[BaseModel],  # noqa
) -> List[BaseModel]:
    """
    Transforms a list of Triton requests into a list of Pydantic objects.

    This function inspects the fields of the given Pydantic type and extracts
    the corresponding tensors from the Triton requests. It then converts these
    tensors into Python types and uses them to instantiate the Pydantic objects.

    Args:
        triton_requests (list): A list of Triton request objects.
        pydantic_type (Type[BaseModel]): The Pydantic model type to which the Triton requests should be transformed.

    Returns:
        list: A list of Pydantic objects instantiated from the Triton requests.

    Raises:
        ValueError: If a tensor corresponding to a Pydantic field cannot be found in the Triton request.
        TypeError: If a tensor corresponding to a Pydantic field has an unsupported type.
    """
    fields = inspect_pydantic_model(pydantic_type)
    results = []

    for request in triton_requests:
        data = {}
        for field in fields:
            field_name, field_type = field.values()
            tensor = get_input_tensor_by_name(request, field_name)
            data[field_name] = convert_tensor_to_python_type(tensor, field_type)
        results.append(pydantic_type(**data))

    return results


def transform_pydantic_to_triton(
    pydantic_objects: List[BaseModel],
) -> List["pb_utils.InferenceResponse"]:
    """
    Transforms a list of Pydantic objects into a list of Triton inference responses.

    This function iterates over the fields of each Pydantic object, determines the
    appropriate tensor data type, and creates a tensor for each field. These tensors
    are then used to create Triton inference responses.

    Args:
        pydantic_objects (list): A list of Pydantic objects to be transformed.

    Returns:
        list: A list of Triton inference responses created from the Pydantic objects.

    Raises:
        ValueError: If a Python type in the Pydantic object does not have a corresponding numpy dtype.
    """
    results = []

    for obj in pydantic_objects:
        tensors = []
        for field, value in obj.dict().items():
            if isinstance(value, list):
                dtype = PYTHON_TYPE_TO_NP_DTYPE[type(value[0])]
            elif isinstance(value, dict):
                dtype = PYTHON_TYPE_TO_NP_DTYPE[type(value)]
                value = [json.dumps(value)]
            else:
                dtype = PYTHON_TYPE_TO_NP_DTYPE[type(value)]
                value = [value]
            tensors.append(create_tensor(field, np.array(value, dtype=dtype)))
        results.append(create_inference_response(tensors))

    return results
