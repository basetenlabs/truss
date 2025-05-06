import contextlib
import importlib.util
import os
import pathlib
from typing import Iterator, Type, TypeVar

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
