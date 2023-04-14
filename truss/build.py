import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable, List, Optional

import click
import cloudpickle
import yaml
from truss.constants import CONFIG_FILE, TEMPLATES_DIR, TRUSS
from truss.docker import kill_containers
from truss.environment_inference.requirements_inference import infer_deps
from truss.errors import FrameworkNotSupportedError
from truss.model_frameworks import MODEL_FRAMEWORKS_BY_TYPE, model_framework_from_model
from truss.model_inference import infer_python_version, map_to_supported_python_version
from truss.notebook import is_notebook_or_ipython
from truss.truss_config import DEFAULT_EXAMPLES_FILENAME, TrussConfig
from truss.truss_handle import TrussHandle
from truss.types import ModelFrameworkType
from truss.util.gpu import get_gpu_memory
from truss.util.path import build_truss_target_directory, copy_file_path, copy_tree_path

logger: logging.Logger = logging.getLogger(__name__)

if is_notebook_or_ipython():
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))


def populate_target_directory(
    config: TrussConfig,
    target_directory_path: Optional[str] = None,
    template: str = "custom",
) -> Path:
    target_directory_path_typed = None
    if target_directory_path is None:
        target_directory_path_typed = build_truss_target_directory(template)
    else:
        target_directory_path_typed = Path(target_directory_path)
        target_directory_path_typed.mkdir(parents=True, exist_ok=True)

    # Create data dir
    (target_directory_path_typed / config.data_dir).mkdir()

    # Create bundled packages dir
    (target_directory_path_typed / config.bundled_packages_dir).mkdir()

    # Create model module dir
    model_dir = target_directory_path_typed / config.model_module_dir
    template_path = TEMPLATES_DIR / template
    copy_tree_path(template_path / "model", model_dir)

    examples_path = template_path / DEFAULT_EXAMPLES_FILENAME
    if examples_path.exists():
        copy_file_path(
            examples_path, target_directory_path_typed / DEFAULT_EXAMPLES_FILENAME
        )

    # Write config
    with (target_directory_path_typed / CONFIG_FILE).open("w") as config_file:
        yaml.dump(config.to_dict(), config_file)

    return target_directory_path_typed


def create_from_model(
    model: Any,
    target_directory: Optional[str] = None,
    data_files: Optional[List[str]] = None,
    requirements_file: Optional[str] = None,
    bundled_packages: Optional[List[str]] = None,
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
        bundled_packages (List[str], optional): Additional local packages that are required by the model.
    Returns:
        TrussHandle: A handle to the generated Truss that provides easy access to content inside.
    """
    model_framework = model_framework_from_model(model)
    if model_framework.typ() == ModelFrameworkType.XGBOOST:
        click.echo(
            click.style(
                """WARNING: Truss uses XGBoost save/load which has a
                different interface during inference than the class
                you used to train this model. You can learn more about
                these differences at
                https://xgboost.readthedocs.io/en/stable/tutorials/saving_model.html
                """,
                fg="yellow",
            )
        )
    if target_directory is None:
        target_directory_path = build_truss_target_directory(
            model_framework.typ().value
        )
    else:
        target_directory_path = Path(target_directory)
    model_framework.to_truss(model, target_directory_path)
    scaf = TrussHandle(target_directory_path)
    _update_truss_props(scaf, data_files, requirements_file, bundled_packages)
    return scaf


def create_from_pipeline(
    pipeline: Callable,
    target_directory: Optional[str] = None,
    data_files: Optional[List[str]] = None,
    requirements_file: Optional[str] = None,
    bundled_packages: Optional[List[str]] = None,
):
    """Create a Truss from a function. A Truss is a build context designed to
    be built as a container locally or uploaded into a model serving environment.

    Args:
        pipeline (a callable function): A function that is expected to be called
        when the Truss server /predict is invoked.
        target_directory (str, optional): The local directory target for the Truss. Otherwise a temporary directory
            will be generated
        data_files (List[str], optional): Additional files required for model operation. Can be a glob that resolves to
            files for the root directory or a directory path.
        requirements_file (str, optional): A file of packages in a PIP requirements format to be installed in the
            container environment.
        bundled_packages (List[str], optional): Additional local packages that are required by the model.
    Returns:
        TrussHandle: A handle to the generated Truss that provides easy access to content inside.
    """

    # Create Truss config
    requirements = list(infer_deps(must_include_deps=set(["cloudpickle"])))
    config = TrussConfig(
        model_type="custom",
        model_framework=ModelFrameworkType.CUSTOM,
        python_version=map_to_supported_python_version(infer_python_version()),
        requirements=requirements,
    )

    # Create and populate target directory path
    target_directory_path = populate_target_directory(
        config=config, target_directory_path=target_directory, template="pipeline"
    )

    gpu_memory = get_gpu_memory()
    if gpu_memory and gpu_memory > 10:
        # TODO: Abu: Remove use of click here
        click.echo(
            click.style(
                """WARNING: Truss identified objects in GPU memory. When serializing a
                function via create(), objects in GPU memory must be moved to
                CPU to be serialized correctly.""",
                fg="yellow",
            )
        )

    # Write Cloudpickled function to data directory
    pipeline_binary_path = target_directory_path / config.data_dir / "pipeline.cpick"
    with open(pipeline_binary_path, "wb") as f:
        cloudpickle.dump(pipeline, f)

    scaf = TrussHandle(target_directory_path)
    _update_truss_props(scaf, data_files, requirements_file, bundled_packages)
    return scaf


def create_from_mlflow_uri(
    model_uri: str,
    target_directory: Optional[str] = None,
    data_files: Optional[List[str]] = None,
    requirements_file: Optional[str] = None,
    bundled_packages: Optional[List[str]] = None,
):
    """Create a Truss with the given model. A Truss is a build context designed to
    be built as a container locally or uploaded into a model serving environment.

    Args:
        model_uri (str): URI pointing to the MLflow model.
        target_directory (str, optional): The local directory target for the Truss. Otherwise a temporary directory
            will be generated
        data_files (List[str], optional): Additional files required for model operation. Can be a glob that resolves to
            files for the root directory or a directory path.
        requirements_file (str, optional): A file of packages in a PIP requirements format to be installed in the
            container environment.
        bundled_packages (List[str], optional): Additional local packages that are required by the model.
    Returns:
        TrussHandle: A handle to the generated Truss that provides easy access to content inside.
    """
    model_framework = MODEL_FRAMEWORKS_BY_TYPE[ModelFrameworkType.MLFLOW]
    if target_directory is None:
        target_directory_path = build_truss_target_directory(
            model_framework.typ().value
        )
    else:
        target_directory_path = Path(target_directory)
    model_framework.to_truss(model_uri, target_directory_path)
    truss = TrussHandle(target_directory_path)
    _update_truss_props(truss, data_files, requirements_file, bundled_packages)
    return truss


def create_from_model_with_exception_handler(*args):
    # returns None if framework not supported, otherwise the Truss
    try:
        return create_from_model(*args)
    except FrameworkNotSupportedError:
        return None


def init(
    target_directory: str,
    data_files: Optional[List[str]] = None,
    requirements_file: Optional[str] = None,
    bundled_packages: Optional[List[str]] = None,
    trainable: bool = False,
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
        model_type="custom",
        model_framework=ModelFrameworkType.CUSTOM,
        python_version=map_to_supported_python_version(infer_python_version()),
    )

    target_directory_path = populate_target_directory(
        config=config, target_directory_path=target_directory
    )

    if trainable:
        _populate_default_training_code(
            config, target_directory_path=Path(target_directory)
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


def create(
    model: Any,
    target_directory: Optional[str] = None,
    data_files: Optional[List[str]] = None,
    requirements_file: Optional[str] = None,
    bundled_packages: Optional[List[str]] = None,
) -> TrussHandle:
    # Some model objects can are callable (like Keras models)
    # so we first attempt to make Truss via a model object

    model_scaffold = create_from_model_with_exception_handler(
        model, target_directory, data_files, requirements_file, bundled_packages
    )
    if model_scaffold:
        return model_scaffold
    else:
        if callable(model):
            return create_from_pipeline(
                model, target_directory, data_files, requirements_file, bundled_packages
            )

    raise ValueError(
        "Invalid input to make Truss. Truss expects a supported framework or callable function."
    )


def mk_truss(*args, **kwargs):
    logger.warn("DeprecationWarning: mk_truss() is deprecated. Use create() instead.")
    return create(*args, **kwargs)


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


def _populate_default_training_code(
    config: TrussConfig,
    target_directory_path: Path,
) -> None:
    """Populate default training code in a truss.

    Assumes target directory already exists.
    """
    # TODO(pankaj): Add support customization based on model framework type, for
    # now we don't support this.
    truss_template = "custom"
    template_path = TEMPLATES_DIR / truss_template
    truss_training_module_dir = target_directory_path / config.train.training_module_dir
    copy_tree_path(template_path / "train", truss_training_module_dir)


def kill_all() -> None:
    kill_containers({TRUSS: True})
