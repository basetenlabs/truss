import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

import yaml

from truss.constants import CONFIG_FILE, TEMPLATES_DIR, TRUSS
from truss.docker import kill_containers
from truss.model_inference import infer_python_version, map_to_supported_python_version
from truss.notebook import is_notebook_or_ipython
from truss.truss_config import Build, TrussConfig
from truss.truss_handle import TrussHandle
from truss.util.path import build_truss_target_directory, copy_tree_path

logger: logging.Logger = logging.getLogger(__name__)

if is_notebook_or_ipython():
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))


def populate_target_directory(
    config: TrussConfig,
    target_directory_path: Optional[str] = None,
    template: str = "custom",
    populate_dirs: bool = True,
) -> Path:
    target_directory_path_typed = None
    if target_directory_path is None:
        target_directory_path_typed = build_truss_target_directory(template)
    else:
        target_directory_path_typed = Path(target_directory_path)
        target_directory_path_typed.mkdir(parents=True, exist_ok=True)

    if populate_dirs:
        # Create data dir
        (target_directory_path_typed / config.data_dir).mkdir()

        # Create bundled packages dir
        # TODO: Drop by default
        (target_directory_path_typed / config.bundled_packages_dir).mkdir()

        # Create model module dir
        model_dir = target_directory_path_typed / config.model_module_dir
        template_path = TEMPLATES_DIR / template
        copy_tree_path(template_path / "model", model_dir)

    # Write config
    with (target_directory_path_typed / CONFIG_FILE).open("w") as config_file:
        yaml.dump(config.to_dict(verbose=False), config_file)

    return target_directory_path_typed


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
    config = TrussConfig(
        model_name=model_name,
        python_version=map_to_supported_python_version(infer_python_version()),
    )

    if build_config:
        config.build = build_config

    target_directory_path = populate_target_directory(
        config=config,
        target_directory_path=target_directory,
        populate_dirs=True,
    )

    scaf = TrussHandle(target_directory_path)
    _update_truss_props(scaf, data_files, requirements_file, bundled_packages)
    return scaf


def load(truss_directory: str) -> TrussHandle:
    """Get a handle to a Truss. A Truss is a build context designed to be built
    as a container locally or uploaded into a model serving environment.

    Args:
        truss_directory (str): The local directory of an existing Truss
    Returns:
        TrussHandle
    """
    return TrussHandle(Path(truss_directory))


def from_directory(*args, **kwargs):
    logger.warn(
        "DeprecationWarning: from_directory() is deprecated. Use load() instead."
    )
    return load(*args, **kwargs)


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


def kill_all() -> None:
    kill_containers({TRUSS: True})
