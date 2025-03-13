import contextlib
import importlib.util
import os
import pathlib
from typing import Iterator

from truss_train import definitions


@contextlib.contextmanager
def import_target(module_path: pathlib.Path) -> Iterator[definitions.TrainingProject]:
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
    training_projects = [
        sym for sym in module_vars if isinstance(sym, definitions.TrainingProject)
    ]

    if len(training_projects) == 0:
        raise ValueError(f"No `{definitions.TrainingProject}` was found.")
    elif len(training_projects) > 1:
        raise ValueError(f"Multiple `{definitions.TrainingProject}`s were found.")

    yield training_projects[0]
