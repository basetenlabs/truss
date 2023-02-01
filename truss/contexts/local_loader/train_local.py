import inspect
from pathlib import Path

from truss.contexts.local_loader.truss_module_loader import truss_module_loaded
from truss.contexts.local_loader.utils import (
    prepare_secrets,
    signature_accepts_keyword_arg,
)
from truss.contexts.truss_context import TrussContext
from truss.truss_spec import TrussSpec


class LocalTrainer(TrussContext):
    """Allows training a truss locally."""

    @staticmethod
    def run(truss_dir: Path):
        def train(variables: dict = None):
            spec = TrussSpec(truss_dir)
            with truss_module_loaded(
                str(truss_dir),
                spec.train_module_fullname,
                spec.bundled_packages_dir.name,
                [str(path.resolve()) for path in spec.external_package_dirs_paths],
            ) as module:
                train_class = getattr(module, spec.train_class_name)
                train_class_signature = inspect.signature(train_class)
                train_init_params = {}
                config = spec.config
                if signature_accepts_keyword_arg(train_class_signature, "config"):
                    train_init_params["config"] = config.to_dict()
                if signature_accepts_keyword_arg(train_class_signature, "output_dir"):
                    train_init_params["output_dir"] = truss_dir / "data"
                if signature_accepts_keyword_arg(train_class_signature, "secrets"):
                    train_init_params["secrets"] = prepare_secrets(spec)

                # Wire up variables
                if signature_accepts_keyword_arg(train_class_signature, "variables"):
                    runtime_variables = {}
                    runtime_variables.update(config.train.variables)
                    if variables is not None:
                        runtime_variables.update(variables)
                    train_init_params["variables"] = runtime_variables
                trainer = train_class(**train_init_params)

                if hasattr(trainer, "pre_train"):
                    trainer.pre_train()

                trainer.train()

                if hasattr(trainer, "post_train"):
                    trainer.post_train()

        return train
