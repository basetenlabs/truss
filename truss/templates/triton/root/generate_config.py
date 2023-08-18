import importlib
from pathlib import Path
from types import ModuleType
from typing import Any, List, Type, Union

import yaml
from jinja2 import Environment, FileSystemLoader, Template
from pydantic import BaseModel

# Copied from truss/templates/triton/model/utils/pydantic.py
FieldAnnotationType = Union[
    Type[int], Type[float], Type[str], Type[bool], Type[dict], List[Any]
]

# Define the mapping from pydantic types to Triton types
TYPE_MAPPING = {
    "str": "TYPE_STRING",
    "int": "TYPE_INT32",
    "float": "TYPE_FP32",
    "bool": "TYPE_BOOL",
    "dict": "TYPE_STRING",
}


class FieldDescriptor:
    def __init__(self, name: str, triton_type: str, dims: List[int]):
        self.name = name
        self.triton_type = triton_type
        self.dims = dims

    @classmethod
    def from_pydantic_field(cls, field_name: str, field_info: Any) -> "FieldDescriptor":
        return cls(
            name=field_name,
            triton_type=get_triton_type(field_info.annotation),
            dims=get_triton_dims(field_info),
        )

    @classmethod
    def from_pydantic_class(
        cls, pydantic_class: Type[BaseModel]
    ) -> List["FieldDescriptor"]:
        return [
            cls.from_pydantic_field(field_name, field_info)
            for field_name, field_info in pydantic_class.__fields__.items()  # type: ignore
        ]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "dims": self.dims,
            "triton_type": self.triton_type,
        }


def read_template_from_fs(base_dir: Path, template_file_name: str) -> Template:
    """Reads a Jinja template from the filesystem."""
    template_loader = FileSystemLoader(str(base_dir))
    template_env = Environment(loader=template_loader)
    return template_env.get_template(template_file_name)


def path_to_module(path: Path) -> str:
    """Convert a filesystem path to a Python module string."""
    return str(path).replace("/", ".")


# Copied from truss/templates/triton/model/utils/pydantic.py
def is_conlist(pydantic_field_annotation: FieldAnnotationType) -> bool:
    """Checks if a Pydantic field annotation is a conlist."""
    # Annotations that are not directly a Python type have an __origin__ attribute
    if not hasattr(pydantic_field_annotation, "__origin__"):
        return False
    return pydantic_field_annotation.__origin__ == list


def get_triton_type(pydantic_field: FieldAnnotationType) -> str:
    if is_conlist(pydantic_field):
        # If the type is a list, get the type of the list elements
        pydantic_field = pydantic_field.__args__[0]  # type: ignore

    type_name = (
        pydantic_field.__name__
        if hasattr(pydantic_field, "__name__")
        else str(pydantic_field)
    )

    if type_name not in TYPE_MAPPING:
        raise TypeError(f"Unsupported type: {type_name}")

    return TYPE_MAPPING[type_name]


def get_triton_dims(pydantic_type: Type) -> List[int]:
    if is_conlist(pydantic_type.annotation):
        dims = next(
            (m for m in pydantic_type.metadata if hasattr(m, "min_length")), None
        )
        if dims and dims.min_length == dims.max_length:
            return [-1, dims.min_length]
        else:
            return [-1, -1]
    return [-1]


def validate_user_model_class(module: ModuleType, required_attributes: List[str]):
    """
    Validates that a user model class has the required attributes.

    Args:
        module (ModuleType): The module containing the user model class.
        required_attributes (list): A list of required attributes.

    Raises:
        AttributeError: If a required attribute is missing.
    """
    for attribute in required_attributes:
        if not hasattr(module, attribute):
            raise AttributeError(
                f"Truss model class is missing {attribute} attribute. For Triton, \
                we require a Pydantic model class that corresponds to your model \
                input and model output."
            )


def generate_config_pbtxt(
    input_class: Type[BaseModel],
    output_class: Type[BaseModel],
    template_path: Path,
    template_name: str = "config.pbtxt.jinja",
    max_batch_size: int = 1,
    num_replicas: int = 1,
    dynamic_batch_delay_ms: int = 0,
    is_gpu: bool = False,
) -> str:
    config_params = {
        "max_batch_size": max_batch_size,
        "num_replicas": num_replicas,
        "is_gpu": is_gpu,
    }
    template = read_template_from_fs(template_path, template_name)
    input_cls_field_descriptors = FieldDescriptor.from_pydantic_class(input_class)
    output_cls_field_descriptors = FieldDescriptor.from_pydantic_class(output_class)
    inputs = [field.to_dict() for field in input_cls_field_descriptors]
    outputs = [field.to_dict() for field in output_cls_field_descriptors]
    if dynamic_batch_delay_ms > 0:
        config_params["dynamic_batching_delay_microseconds"] = dynamic_batch_delay_ms
    return template.render(inputs=inputs, outputs=outputs, **config_params)


def main():
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

    # Determine if GPU is enabled
    config_arguments["is_gpu"] = False
    if "resources" in config:
        if config["resources"].get("use_gpu", False):
            config_arguments["is_gpu"] = True

    # Import model class
    model_module_name = str(Path(model_class_filename).with_suffix(""))
    module = importlib.import_module(
        f"{path_to_module(user_truss_path)}.model.{model_module_name}"
    )

    # Validate model class
    validate_user_model_class(module, [input_cls_name, output_cls_name])

    # Get pydantic model classes
    input_cls = getattr(module, input_cls_name)
    output_cls = getattr(module, output_cls_name)

    # Generate config.pbtxt
    config_pbtxt = generate_config_pbtxt(
        input_cls, output_cls, Path("/app"), **config_arguments
    )

    # Write config.pbtxt to model directory
    with open(model_repository_path / "config.pbtxt", "w") as f:
        f.write(config_pbtxt)


if __name__ == "__main__":
    main()
