from pathlib import Path
from typing import cast

from truss.remote.baseten.remote import BasetenRemote
from truss.remote.remote_factory import RemoteFactory
from truss_train.definitions import TrainingProject
from truss_train.deployment import create_training_job


def push(training_project: TrainingProject, config: Path, remote: str = "baseten"):
    """
    push
    * creates or updates a training_project
    * creates the training_job in the training_project
    * returns a dictionary with the format:
        {
            "training_project": TrainingProject,
            "training_job": TrainingJob,
        }. The definitions of TrainingProject and TrainingJob can be found in the API docs.
        https://docs.baseten.co/reference/training-api/get-training-job
    """
    remote_provider: BasetenRemote = cast(
        BasetenRemote, RemoteFactory.create(remote=remote)
    )
    job_resp = create_training_job(
        remote_provider=remote_provider,
        training_project=training_project,
        config=config,
    )
    return job_resp
