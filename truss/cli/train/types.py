from dataclasses import dataclass
from typing import Optional

from truss_train.definitions import (
    CheckpointList,
    Compute,
    DeployCheckpointsConfig,
    DeployCheckpointsRuntime,
)


@dataclass
class DeployCheckpointArgs:
    project_id: Optional[str]
    job_id: Optional[str]
    deploy_config_path: Optional[str]


class DeployCheckpointsConfigComplete(DeployCheckpointsConfig):
    """
    DeployCheckpointTrussArgs is a dataclass that mirrors DeployCheckpointsConfig but
    removes the optional fileds. This helps provide type safety internal handling.
    """

    checkpoint_details: CheckpointList
    model_name: str
    runtime: DeployCheckpointsRuntime
    compute: Compute
