from pathlib import Path
from typing import cast

from truss.remote.baseten.remote import BasetenRemote
from truss.remote.remote_factory import RemoteFactory
from truss_train.deployment import create_training_job_from_file


def push(config: Path, remote: str = "baseten"):
    """Create or update a training project and create a training job.

    This function performs the following operations:
    1. Creates or updates a training project
    2. Creates a training job within the training project

    Args:
        config: Path to the Python file that defines the training project and job.
        remote: The remote provider to use. Defaults to "baseten".

    Returns:
        dict: A dictionary containing the created training project and job:
            {
                "training_project": TrainingProject,
                "training_job": TrainingJob,
            }

        For detailed definitions of TrainingProject and TrainingJob, see the API docs:
        https://docs.baseten.co/reference/training-api/get-training-job
    """
    remote_provider: BasetenRemote = cast(
        BasetenRemote, RemoteFactory.create(remote=remote)
    )
    return create_training_job_from_file(remote_provider, config)
