import inspect
import logging
import pathlib
import sys
from ast import ClassDef, FunctionDef
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import pkg_resources
from pkg_resources.extern.packaging.requirements import InvalidRequirement
from truss.constants import (
    HUGGINGFACE_TRANSFORMER,
    KERAS,
    LIGHTGBM,
    PYTORCH,
    SKLEARN,
    TENSORFLOW,
    XGBOOST,
)
from truss.errors import FrameworkNotSupportedError

# lists of versions supported by the truss+base_images
PYTHON_VERSIONS = {
    "py37",
    "py38",
    "py39",
}

logger = logging.getLogger(__name__)


def _infer_model_framework(model_class: str):
    model_framework, _, _ = model_class.__module__.partition(".")
    if model_framework == "transformers":
        return HUGGINGFACE_TRANSFORMER
    if model_framework not in {SKLEARN, TENSORFLOW, KERAS, LIGHTGBM, XGBOOST}:
        try:
            import torch

            if issubclass(model_class, torch.nn.Module):
                model_framework = PYTORCH
            else:
                raise FrameworkNotSupportedError(
                    f"Models must be one of "
                    f"{HUGGINGFACE_TRANSFORMER}, {SKLEARN}, "
                    f"{XGBOOST}, {TENSORFLOW}, {PYTORCH} or "
                    f"{LIGHTGBM} "
                )
        except ModuleNotFoundError:
            raise FrameworkNotSupportedError(
                f"Models must be one of "
                f"{HUGGINGFACE_TRANSFORMER}, {SKLEARN}"
                f"{XGBOOST}, {TENSORFLOW}, or {PYTORCH}. "
                f"{LIGHTGBM} "
            )

    return model_framework


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

    Currently, it maps any versions greater than 3.9 to 3.9.

    Args:
        python_version: in the form py[major_version][minor_version] e.g. py39,
        py310
    """
    python_major_version = int(python_version[2:3])
    python_minor_version = int(python_version[3:])

    if python_major_version > 3:
        raise NotImplementedError("Only python version 3 is supported")

    # TODO(pankaj) Add full support for 3.10 and 3.11, this is stop-gap.
    if python_minor_version > 9:
        logger.info(
            f"Mapping python version {python_major_version}.{python_minor_version}"
            " to 3.9, the highest version that Truss currently supports."
        )
        return "py39"

    if python_minor_version < 7:
        logger.info(
            f"Mapping python version {python_major_version}.{python_minor_version}"
            " to 3.7, the lowest version that Truss currently supports."
        )
        return "py37"

    return python_version


def infer_model_information(model: Any) -> ModelBuildStageOne:
    model_class = _model_class(model)
    model_framework = _infer_model_framework(model_class)
    model_type = model_class.__name__

    return ModelBuildStageOne(
        model_type,
        model_framework,
    )


def parse_requirements_file(requirements_file: str) -> dict:
    name_to_req_str = {}
    with pathlib.Path(requirements_file).open() as reqs_file:
        for raw_req in reqs_file.readlines():
            try:
                req = pkg_resources.Requirement.parse(raw_req)
                if req.specifier:
                    name_to_req_str[req.name] = str(req)
                else:
                    name_to_req_str[str(req)] = str(req)
            except InvalidRequirement:
                # there might be pip requirements that do not conform
                raw_req = str(raw_req).strip()
                name_to_req_str[f"custom_{raw_req}"] = raw_req
            except ValueError:
                # can't parse empty lines
                pass

    return name_to_req_str


def _infer_model_init_parameters(model_class: Any) -> Tuple[List, List]:
    full_arg_spec = inspect.getfullargspec(model_class.__init__)
    named_args = full_arg_spec.args[1:]
    number_of_kwargs = full_arg_spec.defaults and len(full_arg_spec.defaults) or 0
    required_args = full_arg_spec.args[1:-number_of_kwargs]
    return named_args, required_args


def _infer_model_init_parameters_ast(model_class_def: ClassDef) -> Tuple[List, List]:
    named_args = []
    required_args = []
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

    if provided_parameters and type(provided_parameters) is not dict:
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
