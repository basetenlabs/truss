from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from truss_train.definitions import (
    CheckpointDetails,
    Compute,
    DeployCheckpointsConfig,
    DeployCheckpointsRuntime,
)


@dataclass
class PrepareCheckpointArgs:
    project_id: Optional[str]
    job_id: Optional[str]
    deploy_config_path: Optional[str]


class DeployCheckpointsConfigComplete(DeployCheckpointsConfig):
    """
    DeployCheckpointTrussArgs is a dataclass that mirrors DeployCheckpointsConfig but
    removes the optional fileds. This helps provide type safety internal handling.
    """

    checkpoint_details: CheckpointDetails
    model_name: str
    deployment_name: str
    runtime: DeployCheckpointsRuntime
    compute: Compute


@dataclass
class PrepareCheckpointResult:
    truss_directory: Path
    checkpoint_deploy_config: DeployCheckpointsConfigComplete
