import importlib
import inspect
import sys
from pathlib import Path

import yaml
from secrets_resolver import SecretsResolver

CONFIG_FILE = "config.yaml"


def _signature_accepts_keyword_arg(signature: inspect.Signature, kwarg: str) -> bool:
    return kwarg in signature.parameters or _signature_accepts_kwargs(signature)


# todo: avoid duplication with model_wrapper
def _signature_accepts_kwargs(signature: inspect.Signature) -> bool:
    for param in signature.parameters.values():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            return True
    return False


def _add_bundled_packages_to_path(config):
    if "bundled_packages_dir" in config:
        bundled_packages_path = Path("/packages")
        if bundled_packages_path.exists():
            sys.path.append(str(bundled_packages_path))


def _load_train_class(config):
    train_module_name = str(Path(config["train_class_filename"]).with_suffix(""))
    train_module = importlib.import_module(
        f"{config['train_module_dir']}.{train_module_name}"
    )
    return getattr(train_module, config["train_class_name"])


def _create_trainer(config):
    train_class = _load_train_class(config)
    train_class_signature = inspect.signature(train_class)
    train_init_params = {}
    if _signature_accepts_keyword_arg(train_class_signature, "config"):
        train_init_params["config"] = config
    if _signature_accepts_keyword_arg(
        train_class_signature, "output_model_artifacts_dir"
    ):
        train_init_params[
            "output_model_artifacts_dir"
        ] = Path()  # todo: wire it up, cwd for now
    if _signature_accepts_keyword_arg(train_class_signature, "secrets"):
        train_init_params["secrets"] = SecretsResolver.get_secrets(config)
    # todo: wire up variables, perhaps through VariablesResolver
    return train_class(**train_init_params)


if __name__ == "__main__":
    with open(CONFIG_FILE, encoding="utf-8") as config_file:
        truss_config = yaml.safe_load(config_file)
        _add_bundled_packages_to_path(truss_config)
        trainer = _create_trainer(truss_config)
        trainer.pre_train()
        trainer.train()
        trainer.post_train()
        # todo: artifact upload
