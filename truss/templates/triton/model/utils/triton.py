import json


def get_numpy_item(tensor):
    item = tensor.as_numpy().item()
    return item.decode("utf-8") if isinstance(item, bytes) else item


def convert_tensor_to_python_type(tensor, dtype):
    if dtype == "dict":
        item = get_numpy_item(tensor)
        return json.loads(item)
    elif dtype == "list":
        return tensor.as_numpy().flatten().tolist()
    elif dtype in ["str", "int", "float", "bool"]:
        return get_numpy_item(tensor)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def get_input_tensor_by_name(obj, name):
    """Extracts a tensor from a Triton InferenceRequest by name."""
    from triton_python_backend_utils import get_input_tensor_by_name

    x = get_input_tensor_by_name(obj, name)
    if x is None:
        raise ValueError(f"Input tensor {name} not found in request.")
    return x


def create_tensor(name, value):
    """Creates a Triton tensor."""
    from triton_python_backend_utils import Tensor

    return Tensor(name, value)


def create_inference_response(objects):
    """Creates a Triton InferenceResponse from a list of Triton tensors."""
    from triton_python_backend_utils import InferenceResponse

    return InferenceResponse(objects)
