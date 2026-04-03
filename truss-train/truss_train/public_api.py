from pathlib import Path
from typing import Optional, Union, cast, overload

from truss.remote.baseten.remote import BasetenRemote
from truss.remote.remote_factory import RemoteFactory
from truss_train import loader
from truss_train.definitions import TrainingProject
from truss_train.deployment import _upsert_project_and_create_job, create_training_job


@overload
def push(config: Path, *, remote: str = "baseten") -> dict: ...


@overload
def push(
    config: TrainingProject,
    *,
    remote: str = "baseten",
    source_dir: Optional[Path] = None,
) -> dict: ...


def push(
    config: Union[Path, TrainingProject],
    *,
    remote: str = "baseten",
    source_dir: Optional[Path] = None,
) -> dict:
    """Create or update a training project and create a training job.

    Args:
        config: Either a Path to a Python config file that defines the training
            project and job, or a TrainingProject instance constructed in code.
        remote: The remote provider to use. Defaults to "baseten".
        source_dir: Base directory for workspace path resolution and archiving.
            Only used when config is a TrainingProject. Defaults to Path.cwd().

    Returns:
        dict: A dictionary containing the created training project and job.
    """
    remote_provider: BasetenRemote = cast(
        BasetenRemote, RemoteFactory.create(remote=remote)
    )

    if isinstance(config, Path):
        with loader.import_training_project(config) as training_project:
            return create_training_job(remote_provider, config, training_project)
    else:
        resolved_source_dir = source_dir if source_dir is not None else Path.cwd()
        return _upsert_project_and_create_job(
            remote_provider, config, resolved_source_dir
        )
