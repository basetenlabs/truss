import inspect
from pathlib import Path

from truss.contexts.local_loader.truss_module_loader import truss_module_loaded
from truss.contexts.local_loader.utils import (
    prepare_secrets,
    signature_accepts_keyword_arg,
)
from truss.contexts.truss_context import TrussContext
from truss.truss_spec import TrussSpec


class LoadModelLocal(TrussContext):
    """Loads a Truss model locally.

    The loaded model can be used to make predictions for quick testing.
    Runs in the current pip environment directly. Assumes all requirements and
    system packages are already installed.
    """

    @staticmethod
    def run(truss_dir: Path):
        spec = TrussSpec(truss_dir)
        with truss_module_loaded(
            str(truss_dir),
            spec.model_module_fullname,
            spec.bundled_packages_dir.name,
            [str(path.resolve()) for path in spec.external_package_dirs_paths],
        ) as module:
            model_class = getattr(module, spec.model_class_name)
            model_class_signature = inspect.signature(model_class)
            model_init_params = {}
            if signature_accepts_keyword_arg(model_class_signature, "config"):
                model_init_params["config"] = spec.config.to_dict()
            if signature_accepts_keyword_arg(model_class_signature, "data_dir"):
                model_init_params["data_dir"] = truss_dir / "data"
            if signature_accepts_keyword_arg(model_class_signature, "secrets"):
                model_init_params["secrets"] = prepare_secrets(spec)
            model = model_class(**model_init_params)
            if hasattr(model, "load"):
                model.load()
            return model
