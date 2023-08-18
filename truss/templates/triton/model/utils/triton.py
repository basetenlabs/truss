import json
from typing import Any, List

import triton_python_backend_utils as pb_utils


def get_numpy_item(
    tensor: pb_utils.Tensor,
) -> str:  # noqa # type: ignore
    """Returns the item of a numpy array as a Python type."""
    item = tensor.as_numpy().item()
    return item.decode("utf-8") if isinstance(item, bytes) else item


def convert_tensor_to_python_type(tensor: pb_utils.Tensor, dtype: str) -> Any:
    """Converts a Triton tensor to a Python type."""

    def _convert_tensor_to_dict(
        tensor: pb_utils.Tensor,
    ) -> dict:
        try:
            item = json.loads(get_numpy_item(tensor))
        except json.decoder.JSONDecodeError:
            raise ValueError(
                f"Could not decode string to dict: {item}. Please ensure that the string is valid JSON."
            )
        return item

    if dtype == "dict":
        return _convert_tensor_to_dict(tensor)
    elif dtype == "list":
        return tensor.as_numpy().flatten().tolist()
    elif dtype in ["str", "int", "float", "bool"]:
        return get_numpy_item(tensor)
    else:
        raise TypeError(f"Unsupported type: {dtype}")


def get_input_tensor_by_name(
    obj: pb_utils.InferenceRequest, name: str  # noqa
) -> pb_utils.Tensor:  # noqa
    """Extracts a tensor from a Triton InferenceRequest by name."""
    x = pb_utils.get_input_tensor_by_name(obj, name)
    if x is None:
        raise ValueError(f"Input tensor {name} not found in request.")
    return x


def create_tensor(name: str, value: Any) -> pb_utils.Tensor:  # noqa
    """Creates a Triton tensor."""
    return pb_utils.Tensor(name, value)


def create_inference_response(
    objects: List[pb_utils.Tensor],
) -> pb_utils.InferenceResponse:
    """Creates a Triton InferenceResponse from a list of Triton tensors."""
    return pb_utils.InferenceResponse(objects)
