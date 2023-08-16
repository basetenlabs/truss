import json

import numpy as np
from utils.pydantic import PYTHON_TYPE_TO_NP_DTYPE, inspect_pydantic_model
from utils.triton import (
    convert_tensor_to_python_type,
    create_inference_response,
    create_tensor,
    get_input_tensor_by_name,
)


def transform_triton_to_pydantic(triton_requests, pydantic_type):
    fields = inspect_pydantic_model(pydantic_type)
    print("Fields from the Pydantic model:", fields)
    results = []

    for request in triton_requests:
        data = {}
        for field in fields:
            field_name, field_type = field.values()
            tensor = get_input_tensor_by_name(request, field_name)
            print("Pre converted field", tensor)
            data[field_name] = convert_tensor_to_python_type(tensor, field_type)
            print("Post converted field", data[field_name])
        results.append(pydantic_type(**data))

    return results


def transform_pydantic_to_triton(pydantic_objects):
    results = []

    for obj in pydantic_objects:
        tensors = []
        for field, value in obj.dict().items():
            print("Field", field)
            print("Value", value)
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
