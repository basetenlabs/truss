from dataclasses import dataclass
from typing import Optional

from pydantic import BaseModel

from truss.base import truss_config
from truss_train.definitions import (
    CheckpointList,
    Compute,
    DeployCheckpointsConfig,
    DeployCheckpointsRuntime,
)


@dataclass
class DeployCheckpointArgs:
    dry_run: bool
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


class DeploySuccessModelVersion(BaseModel):
    # allow extra fields to be forwards compatible with server
    class Config:
        extra = "allow"

    name: str
    id: str


class DeploySuccessResult(BaseModel):
    deploy_config: DeployCheckpointsConfigComplete
    truss_config: Optional[truss_config.TrussConfig]
    model_version: Optional[DeploySuccessModelVersion]
