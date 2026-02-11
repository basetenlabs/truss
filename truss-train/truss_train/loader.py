import contextlib
import importlib.util
import os
import pathlib
from typing import Iterator, Type, TypeVar

import yaml

from truss_train import definitions

T = TypeVar("T")


@contextlib.contextmanager
def import_training_project(
    module_path: pathlib.Path,
) -> Iterator[definitions.TrainingProject]:
    with import_target(module_path, definitions.TrainingProject) as project:
        yield project


@contextlib.contextmanager
def import_deploy_checkpoints_config(
    module_path: pathlib.Path,
) -> Iterator[definitions.DeployCheckpointsConfig]:
    with import_target(module_path, definitions.DeployCheckpointsConfig) as config:
        yield config


@contextlib.contextmanager
def import_auto_sft(
    module_path: pathlib.Path,
) -> Iterator[definitions.AutoSFT]:
    with import_target(module_path, definitions.AutoSFT) as config:
        yield config


def load_auto_sft_from_yaml(yaml_path: pathlib.Path) -> definitions.AutoSFT:
    """Load AutoSFT config from a YAML file."""
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    if not data:
        raise ValueError(f"Config file {yaml_path} is empty or invalid.")
    # PyYAML parses values like 2e-5 as strings; coerce learning_rate to float
    if "learning_rate" in data and isinstance(data["learning_rate"], str):
        try:
            data["learning_rate"] = float(data["learning_rate"])
        except ValueError:
            pass
    return definitions.AutoSFT.model_validate(data)


@contextlib.contextmanager
def import_target(module_path: pathlib.Path, target_type: Type[T]) -> Iterator[T]:
    module_name = module_path.stem
    if not os.path.isfile(module_path):
        raise ImportError(
            f"`{module_path}` is not a file. You must point to a python file where "
            f"the training configuration is defined."
        )

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if not spec or not spec.loader:
        raise ImportError(f"Could not import `{module_path}`. Check path.")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    module_vars = (getattr(module, name) for name in dir(module))
    target = [sym for sym in module_vars if isinstance(sym, target_type)]

    if len(target) == 0:
        raise ValueError(f"No `{target_type}` was found.")
    elif len(target) > 1:
        raise ValueError(f"Multiple `{target_type}`s were found.")

    yield target[0]
