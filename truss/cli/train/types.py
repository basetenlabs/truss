from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from truss_train.definitions import (
    CheckpointDeployConfig,
    CheckpointDeployRuntime,
    CheckpointDetails,
    Compute,
)


@dataclass
class PrepareCheckpointArgs:
    project_id: Optional[str]
    job_id: Optional[str]
    deploy_config_path: Optional[str]


@dataclass
class PrepareCheckpointResult:
    truss_directory: Path
    checkpoint_deploy_config: CheckpointDeployConfig


@dataclass
class DeployCheckpointTrussArgs(CheckpointDeployConfig):
    """
    DeployCheckpointTrussArgs is a dataclass that mirrors CheckpointDeployConfig but
    removes the optional fileds. This helps provide type safety internal handling.
    """

    checkpoint_details: CheckpointDetails
    model_name: str
    base_model_id: str
    deployment_name: str
    runtime: CheckpointDeployRuntime
    compute: Compute
