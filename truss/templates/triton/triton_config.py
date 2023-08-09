import importlib
from pathlib import Path
from typing import List, Type

import yaml
from jinja2 import Environment, FileSystemLoader, Template
from pydantic import BaseModel, conlist

# Define the mapping from pydantic types to Triton types
TYPE_MAPPING = {
    "str": "TYPE_STRING",
    "int": "TYPE_INT32",
    "float": "TYPE_FP32",
    "bool": "TYPE_BOOL",
    "dict": "TYPE_STRING",
    "conlist": "TYPE_FP32",
    "list": "TYPE_FP32",
}


def read_template_from_fs(base_dir: Path, template_file_name: str) -> Template:
    template_loader = FileSystemLoader(str(base_dir))
    template_env = Environment(loader=template_loader)
    return template_env.get_template(template_file_name)


def path_to_module(path: Path) -> str:
    """Convert a filesystem path to a Python module string."""
    return str(path).replace("/", ".")


def is_list_type(pydantic_type: Type) -> bool:
    return pydantic_type == List or (
        hasattr(pydantic_type, "__origin__") and pydantic_type.__origin__ == list
    )


def is_conlist_type(pydantic_type: Type) -> bool:
    return hasattr(pydantic_type, "__origin__") and pydantic_type.__origin__ == conlist


def get_triton_type(pydantic_type: Type) -> str:
    if is_list_type(pydantic_type):
        return TYPE_MAPPING["list"]
    elif is_conlist_type(pydantic_type):
        return TYPE_MAPPING["conlist"]
    else:
        return TYPE_MAPPING[pydantic_type.__name__]


def get_dims(pydantic_type: Type) -> List[int]:
    if is_list_type(pydantic_type):
        return [-1]
    elif is_conlist_type(pydantic_type):
        # Assuming min_items and max_items are the same for fixed-length lists
        return [pydantic_type.__args__[1].min_items]
    else:
        return [1]


def generate_config_pbtxt(
    input_class: BaseModel,
    output_class: BaseModel,
    template_path: Path,
    template_name: str = "config.pbtxt.jinja",
) -> str:
    inputs = [
        {
            "name": name,
            "type": get_triton_type(input_class.__annotations__[name]),
            "dims": get_dims(input_class.__annotations__[name]),
        }
        for name in input_class.model_fields.keys()  # type: ignore
    ]
    outputs = [
        {
            "name": name,
            "type": get_triton_type(output_class.__annotations__[name]),
            "dims": get_dims(output_class.__annotations__[name]),
        }
        for name in output_class.model_fields.keys()  # type: ignore
    ]

    return read_template_from_fs(template_path, template_name).render(
        inputs=inputs, outputs=outputs
    )


if __name__ == "__main__":
    model_repository_path = Path("model")
    user_truss_path = model_repository_path / "1" / "truss"

    with open(user_truss_path / "config.yaml", "r") as f:
        config = yaml.safe_load(f)

    model_class_filename = config["model_class_filename"]
    input_cls_name = config.get("input_type_name", "Input")
    output_cls_name = config.get("output_type_name", "Output")

    # Import model class
    model_module_name = str(Path(model_class_filename).with_suffix(""))
    module = importlib.import_module(
        f"{path_to_module(user_truss_path)}.model.{model_module_name}"
    )

    try:
        input_cls = getattr(module, input_cls_name)
        output_cls = getattr(module, output_cls_name)
    except AttributeError:
        raise AttributeError(
            f"Model class {model_class_filename} is missing {input_cls_name} or {output_cls_name} class."
        )

    # Generate config.pbtxt
    config_pbtxt = generate_config_pbtxt(input_cls, output_cls, Path("/app"))

    # Write config.pbtxt to model directory
    with open(model_repository_path / "config.pbtxt", "w") as f:
        f.write(config_pbtxt)
