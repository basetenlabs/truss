import importlib
from pathlib import Path
from typing import List, Type

import yaml
from jinja2 import Environment, FileSystemLoader, Template
from pydantic import BaseModel

# Define the mapping from pydantic types to Triton types
TYPE_MAPPING = {
    "str": "TYPE_STRING",
    "int": "TYPE_INT32",
    "float": "TYPE_FP32",
    "bool": "TYPE_BOOL",
    "dict": "TYPE_STRING",
}


def read_template_from_fs(base_dir: Path, template_file_name: str) -> Template:
    template_loader = FileSystemLoader(str(base_dir))
    template_env = Environment(loader=template_loader)
    return template_env.get_template(template_file_name)


def path_to_module(path: Path) -> str:
    """Convert a filesystem path to a Python module string."""
    return str(path).replace("/", ".")


def is_list_type(pydantic_field) -> bool:
    actual_type = getattr(pydantic_field, "outer_type_", pydantic_field)
    return actual_type == List or (
        hasattr(actual_type, "__origin__") and actual_type.__origin__ == list
    )


def get_triton_type(pydantic_field: Type) -> str:
    if is_list_type(pydantic_field):
        # Assuming the first argument of the list type is the base type
        base_type = pydantic_field.__args__[0]
    else:
        base_type = pydantic_field

    # Determine the name of the base type
    type_name = (
        base_type.__name__
        if hasattr(base_type, "__name__")
        else base_type.__origin__.__name__
    )

    if type_name not in TYPE_MAPPING:
        raise ValueError(f"Unsupported type: {type_name}")

    return TYPE_MAPPING[type_name]


def get_dims(pydantic_type: Type) -> List[int]:
    if is_list_type(pydantic_type.annotation):
        dims = next(
            (m for m in pydantic_type.metadata if hasattr(m, "min_length")), None
        )
        if dims and dims.min_length == dims.max_length:
            return [-1, dims.min_length]
        else:
            return [-1, -1]
    return [-1]


def generate_config_pbtxt(
    input_class: BaseModel,
    output_class: BaseModel,
    template_path: Path,
    template_name: str = "config.pbtxt.jinja",
    max_batch_size: int = 1,
    num_replicas: int = 1,
    dynamic_batch_delay_ms: int = 0,
) -> str:
    def _inspect_pydantic_model(pydantic_cls):
        return [
            {
                "name": field_name,
                "type": get_triton_type(field_info.annotation),
                "dims": get_dims(field_info),
            }
            for field_name, field_info in pydantic_cls.model_fields.items()
        ]

    config_params = {
        "max_batch_size": max_batch_size,
        "num_replicas": num_replicas,
    }
    inputs = _inspect_pydantic_model(input_class)
    outputs = _inspect_pydantic_model(output_class)
    template = read_template_from_fs(template_path, template_name)
    if dynamic_batch_delay_ms > 0:
        config_params["dynamic_batching_delay_microseconds"] = dynamic_batch_delay_ms
    return template.render(inputs=inputs, outputs=outputs, **config_params)


if __name__ == "__main__":
    model_repository_path = Path("model")
    user_truss_path = model_repository_path / "1" / "truss"

    with open(user_truss_path / "config.yaml", "r") as f:
        config = yaml.safe_load(f)

    build = config.get("build", {})
    build.pop("model_server")
    config_arguments = {k: v for d in build["arguments"] for k, v in d.items()}

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
    config_pbtxt = generate_config_pbtxt(
        input_cls, output_cls, Path("/app"), **config_arguments
    )

    # Write config.pbtxt to model directory
    with open(model_repository_path / "config.pbtxt", "w") as f:
        f.write(config_pbtxt)
