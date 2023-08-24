import json
from typing import Any, List

import numpy as np
from utils.errors import UnsupportedTypeError


class Tensor:
    def as_numpy(self) -> np.ndarray:
        return  # type: ignore


class InferenceRequest:
    pass


class InferenceResponse:
    pass


def get_numpy_item(
    tensor: Tensor,
) -> str:
    """Returns the item of a numpy array as a Python type."""
    item = tensor.as_numpy().item()
    return item.decode("utf-8") if isinstance(item, bytes) else item


def convert_tensor_to_python_type(tensor: Tensor, dtype: str) -> Any:
    """Converts a Triton tensor to a Python type."""

    def _convert_tensor_to_dict(
        tensor: Tensor,
    ) -> dict:
        item = get_numpy_item(tensor)
        try:
            coerced_dict = json.loads(item)
        except json.decoder.JSONDecodeError:
            raise ValueError(
                f"Could not decode string to dict: {item}. Please ensure that the string is valid JSON."
            )
        return coerced_dict

    if dtype == "dict":
        return _convert_tensor_to_dict(tensor)
    elif dtype == "list":
        return tensor.as_numpy().flatten().tolist()
    elif dtype in ["str", "int", "float", "bool"]:
        return get_numpy_item(tensor)
    else:
        raise UnsupportedTypeError(f"Unsupported dtype: {dtype}")


def get_input_tensor_by_name(obj: InferenceRequest, name: str) -> Tensor:
    """Extracts a tensor from a Triton InferenceRequest by name."""
    from triton_python_backend_utils import get_input_tensor_by_name

    x = get_input_tensor_by_name(obj, name)
    if x is None:
        raise ValueError(f"Input tensor {name} not found in request.")
    return x


def create_tensor(name: str, value: Any) -> Tensor:
    """Creates a Triton tensor."""
    from triton_python_backend_utils import Tensor

    return Tensor(name, value)


def create_inference_response(
    objects: List[Tensor],
) -> InferenceResponse:
    """Creates a Triton InferenceResponse from a list of Triton tensors."""
    from triton_python_backend_utils import InferenceResponse

    return InferenceResponse(objects)
