import os
from pathlib import Path
from typing import Any, List

import click
import yaml

from truss.constants import CONFIG_FILE, TEMPLATES_DIR, TRUSS
from truss.docker import kill_containers
from truss.model_frameworks import model_framework_from_model
from truss.model_inference import infer_python_version
from truss.truss_config import DEFAULT_EXAMPLES_FILENAME, TrussConfig
from truss.truss_handle import TrussHandle
from truss.types import ModelFrameworkType
from truss.utils import (build_truss_target_directory, copy_file_path,
                         copy_tree_path)


def mk_truss(
    model: Any,
    target_directory: str = None,
    data_files: List[str] = None,
    requirements_file: str = None,
) -> TrussHandle:
    """Create a Truss with the given model. A Truss is a build context designed to
    be built as a container locally or uploaded into a model serving environment.

    Args:
        model (an in-memory model object): A model object to be deployed (e.g. a keras, sklearn, or pytorch model
            object)
        target_directory (str, optional): The local directory target for the Truss. Otherwise a temporary directory
            will be generated
        data_files (List[str], optional): Additional files required for model operation. Can be a glob that resolves to
            files for the root directory or a directory path.
        requirements_file (str, optional): A file of packages in a PIP requirements format to be installed in the
            container environment.
    Returns:
        TrussHandle: A handle to the generated Truss that provides easy access to content inside.
    """
    model_framework = model_framework_from_model(model)
    if (model_framework.typ() == ModelFrameworkType.XGBOOST):
        click.echo(
            click.style
            (
                '''WARNING: Truss uses XGBoost save/load which has a
                different interface during inference than the class
                you used to train this model. You can learn more about
                these differences at
                https://xgboost.readthedocs.io/en/stable/tutorials/saving_model.html
                ''', fg='yellow'
            )
        )
    if target_directory is None:
        target_directory_path = build_truss_target_directory(model_framework.typ().value)
    else:
        target_directory_path = Path(target_directory)
    model_framework.to_truss(model, target_directory_path)
    scaf = TrussHandle(target_directory_path)
    _update_truss_props(scaf, data_files, requirements_file)
    return scaf


def init(
    target_directory: str,
    data_files: List[str] = None,
    requirements_file: str = None,
) -> TrussHandle:
    """
    Initialize an empty placeholder Truss. A Truss is a build context designed
    to be built as a container locally or uploaded into a baseten serving
    environment. This placeholder structure can be filled to represent ML
    models.

    Args:
        target_directory: Absolute or relative path of the directory to create
                          Truss in. The directory is created if it doesn't exist.
    """
    target_directory_path = Path(target_directory)
    target_directory_path.mkdir(parents=True, exist_ok=True)
    config = TrussConfig(
        model_type='custom',
        model_framework=ModelFrameworkType.CUSTOM,
        python_version=infer_python_version(),
    )

    # Create data dir
    (target_directory_path / config.data_dir).mkdir()

    # Create model module dir
    model_dir = target_directory_path / config.model_module_dir
    template_path = TEMPLATES_DIR / 'custom'
    copy_tree_path(template_path / 'model', model_dir)

    examples_path = template_path / DEFAULT_EXAMPLES_FILENAME
    if examples_path.exists():
        copy_file_path(examples_path, target_directory_path / DEFAULT_EXAMPLES_FILENAME)

    # Write config
    with (target_directory_path / CONFIG_FILE).open('w') as config_file:
        yaml.dump(config.to_dict(), config_file)

    scaf = TrussHandle(target_directory_path)
    _update_truss_props(scaf, data_files, requirements_file)
    return scaf


def from_directory(truss_directory: str) -> TrussHandle:
    """Get a handle to a Truss. A Truss is a build context designed to be built
       as a container locally or uploaded into a model serving environment.

       Args:
           truss_directory (str): The local directory of an existing Truss
       Returns:
           TrussHandle
       """
    return TrussHandle(Path(truss_directory))


def cleanup():
    """
    Cleans up .truss directory.
    """
    build_folder_path = Path(
        Path.home(),
        '.truss'
    )
    if (build_folder_path.exists()):
        for obj in build_folder_path.glob('**/*'):
            if (not obj.name == 'config.yaml') and (obj.is_file()):
                os.remove(obj)
    return


def _update_truss_props(
    scaf: TrussHandle,
    data_files: List[str] = None,
    requirements_file: str = None,
):
    if data_files is not None:
        for data_file in data_files:
            scaf.add_data(data_file)

    if requirements_file is not None:
        scaf.update_requirements_from_file(requirements_file)


def kill_all():
    kill_containers({
        TRUSS: True
    })
