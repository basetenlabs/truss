import inspect
import logging
import sys
from ast import ClassDef, FunctionDef
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class ModelBuildStageOne:
    # the Python Class of the model
    model_type: str
    # the framework that the model is built in
    model_framework: str


def _model_class(model: Any):
    return model.__class__


def infer_python_version() -> str:
    return f"py{sys.version_info.major}{sys.version_info.minor}"


def map_to_supported_python_version(python_version: str) -> str:
    """Map python version to truss supported python version.

    Currently, it maps any versions greater than 3.11 to 3.11.

    Args:
        python_version: in the form py[major_version][minor_version] e.g. py39,
        py310
    """
    python_major_version = int(python_version[2:3])
    python_minor_version = int(python_version[3:])

    if python_major_version != 3:
        raise NotImplementedError("Only python version 3 is supported")

    if python_minor_version > 11:
        logger.info(
            f"Mapping python version {python_major_version}.{python_minor_version}"
            " to 3.11, the highest version that Truss currently supports."
        )
        return "py311"

    if python_minor_version < 8:
        # TODO: consider raising an error instead - it doesn't' seem safe.
        logger.info(
            f"Mapping python version {python_major_version}.{python_minor_version}"
            " to 3.8, the lowest version that Truss currently supports."
        )
        return "py38"

    return python_version


def _infer_model_init_parameters(model_class: Any) -> Tuple[List, List]:
    full_arg_spec = inspect.getfullargspec(model_class.__init__)
    named_args = full_arg_spec.args[1:]
    number_of_kwargs = full_arg_spec.defaults and len(full_arg_spec.defaults) or 0
    required_args = full_arg_spec.args[1:-number_of_kwargs]
    return named_args, required_args


def _infer_model_init_parameters_ast(model_class_def: ClassDef) -> Tuple[List, List]:
    named_args: List[str] = []
    required_args: List[str] = []
    init_model_functions = [
        node
        for node in model_class_def.body
        if isinstance(node, FunctionDef) and node.name == "__init__"
    ]

    if not init_model_functions:
        return named_args, required_args

    assert (
        len(init_model_functions) == 1
    ), "There should only be one __init__ function in the model class"
    init_model_function = init_model_functions[0]
    named_args = [arg.arg for arg in init_model_function.args.args][1:]
    number_of_defaults = len(init_model_function.args.defaults)
    required_args = named_args[:-number_of_defaults]
    return named_args, required_args


def validate_provided_parameters_with_model(
    model_class: Any, provided_parameters: Dict[str, Any]
) -> None:
    """
    Validates that all provided parameters match the signature of the model.

    Args:
        model_class: The model class to validate against
        provided_parameters: The parameters to validate
    """
    if type(model_class) == ClassDef:
        named_args, required_args = _infer_model_init_parameters_ast(model_class)
    else:
        named_args, required_args = _infer_model_init_parameters(model_class)

    # Check that there are no extra parameters
    if not named_args:
        return

    if provided_parameters and not isinstance(provided_parameters, dict):
        raise TypeError(
            f"Provided parameters must be a dict, not {type(provided_parameters)}"
        )

    for arg in provided_parameters:
        if arg not in named_args:
            raise ValueError(
                f"Provided parameter {arg} is not a valid init parameter for the model."
            )

    for arg in required_args:
        if arg not in provided_parameters:
            raise ValueError(
                f"Required init parameter {arg} was not provided for this model."
            )
