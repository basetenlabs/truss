import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Union

import yaml

from truss.base.constants import (
    CONFIG_FILE,
    PYTHON_DX_CUSTOM_TEMPLATE_DIR,
    TRADITIONAL_CUSTOM_TEMPLATE_DIR,
)
from truss.base.truss_config import (
    Build,
    TrussConfig,
    map_local_to_supported_python_version,
)
from truss.truss_handle.truss_handle import TrussHandle
from truss.util.notebook import is_notebook_or_ipython
from truss.util.path import copy_tree_path

logger: logging.Logger = logging.getLogger(__name__)

if is_notebook_or_ipython():
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))


def _populate_target_directory(
    config: TrussConfig, target_directory: str, python_dx: bool = False
) -> Path:
    target_directory_path = Path(target_directory)
    target_directory_path.mkdir(parents=True, exist_ok=True)

    if not python_dx:
        # Create data dir
        (target_directory_path / config.data_dir).mkdir()

        # Create bundled packages dir
        # TODO: Drop by default
        (target_directory_path / config.bundled_packages_dir).mkdir()

        # Create model module dir
        model_dir = target_directory_path / config.model_module_dir
        copy_tree_path(TRADITIONAL_CUSTOM_TEMPLATE_DIR / "model", model_dir)

        # Write config
        with (target_directory_path / CONFIG_FILE).open("w") as config_file:
            yaml.dump(config.to_dict(verbose=False), config_file)
    else:
        copy_tree_path(PYTHON_DX_CUSTOM_TEMPLATE_DIR, target_directory_path)

        # Hack: We want to place the user provided model name into generated code. Until
        # this gets more complicated, we rely on a brittle string replace. Eventually, we
        # can consider moving to jinja templates.
        model_file_path = target_directory_path / "my_model.py"
        with open(model_file_path, "r") as f:
            content = f.read()

        assert config.model_name is not None
        content = content.replace("{{ MODEL_NAME }}", config.model_name)
        with open(model_file_path, "w") as f:
            f.write(content)

    return target_directory_path


def init_directory(
    target_directory: str,
    build_config: Optional[Build] = None,
    model_name: Optional[str] = None,
    python_dx: bool = False,
) -> Path:
    config = TrussConfig(
        model_name=model_name, python_version=map_local_to_supported_python_version()
    )

    if build_config:
        config.build = build_config

    target_directory_path = _populate_target_directory(
        config=config, target_directory=target_directory, python_dx=python_dx
    )

    return target_directory_path


def init(
    target_directory: str,
    data_files: Optional[List[str]] = None,
    requirements_file: Optional[str] = None,
    bundled_packages: Optional[List[str]] = None,
    build_config: Optional[Build] = None,
    model_name: Optional[str] = None,
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
    target_path = init_directory(
        target_directory=target_directory,
        build_config=build_config,
        model_name=model_name,
    )

    scaf = TrussHandle(target_path)
    _update_truss_props(scaf, data_files, requirements_file, bundled_packages)
    return scaf


def load(truss_directory: Union[str, Path]) -> TrussHandle:
    """Get a handle to a Truss. A Truss is a build context designed to be built
    as a container locally or uploaded into a model serving environment.

    Args:
        truss_directory (str | Path): The local directory of an existing Truss
    Returns:
        TrussHandle
    """
    return TrussHandle(Path(truss_directory))


def cleanup() -> None:
    """
    Cleans up .truss directory.
    """
    build_folder_path = Path(Path.home(), ".truss")
    if build_folder_path.exists():
        for obj in build_folder_path.glob("**/*"):
            if (not obj.name == "config.yaml") and (obj.is_file()):
                os.remove(obj)
    return


def _update_truss_props(
    scaf: TrussHandle,
    data_files: Optional[List[str]] = None,
    requirements_file: Optional[str] = None,
    bundled_packages: Optional[List[str]] = None,
) -> None:
    if data_files is not None:
        for data_file in data_files:
            scaf.add_data(data_file)

    if bundled_packages is not None:
        for package in bundled_packages:
            scaf.add_bundled_package(package)

    if requirements_file is not None:
        scaf.update_requirements_from_file(requirements_file)
