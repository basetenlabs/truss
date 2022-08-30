from pathlib import Path
from typing import Dict, Set

from truss.constants import PYTORCH_REQ_MODULE_NAMES
from truss.model_framework import ModelFramework
from truss.types import ModelFrameworkType

TORCH_PACKAGE_FILE = "model_package.pt"
TORCH_MODEL_PICKLE_FILENAME = "model.pkl"
TORCH_MODEL_PACKAGE_NAME = "torch_model"


class PyTorch(ModelFramework):
    def typ(self) -> ModelFrameworkType:
        return ModelFrameworkType.PYTORCH

    def required_python_depedencies(self) -> Set[str]:
        return PYTORCH_REQ_MODULE_NAMES

    def serialize_model_to_directory(self, model, target_directory: Path):
        from torch import package

        try:
            _torch_package(model, target_directory / TORCH_PACKAGE_FILE, [])
        except package.package_exporter.PackagingError as pkg_err:
            # Make sure previous, potentially partially written, package is gone
            (target_directory / TORCH_PACKAGE_FILE).unlink()
            modules_to_extern = _broken_modules(pkg_err)
            _torch_package(
                model, target_directory / TORCH_PACKAGE_FILE, modules_to_extern
            )

    def model_metadata(self, model) -> Dict[str, str]:
        return {
            "torch_package_file": TORCH_PACKAGE_FILE,
            "torch_model_pickle_filename": TORCH_MODEL_PICKLE_FILENAME,
            "torch_model_package_name": TORCH_MODEL_PACKAGE_NAME,
            "model_binary_dir": "model",
        }

    def supports_model_class(self, model_class) -> bool:
        try:
            import torch

            return issubclass(model_class, torch.nn.Module)
        except ModuleNotFoundError:
            return False


def _torch_package(model, path: Path, extern_modules: list):
    from torch import package

    with package.PackageExporter(path) as exp:
        exp.intern(f"{model.__class__.__module__}.**")
        for extern_module in extern_modules:
            exp.extern(f"{extern_module}.**")
        exp.save_pickle(TORCH_MODEL_PACKAGE_NAME, TORCH_MODEL_PICKLE_FILENAME, model)


def _broken_modules(pkg_error):
    """Extract the broken modules from the torch package error.

    We would extern these modules.
    Args:
        pkg_error (package.package_exporter.PackagingError): Error to extract broken modules from.
    """
    return [
        module_name
        for module_name, attrs in pkg_error.dependency_graph.nodes.items()
        if attrs.get("error") is not None
    ]
