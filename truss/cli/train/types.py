from dataclasses import dataclass, field
from typing import List, Optional

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
    run_id: Optional[str]
    deploy_config_path: Optional[str]
    is_loops_command: bool = False
    # Loops-only: explicit checkpoint PKs (e.g. `tcp_step100`) provided via
    # `--checkpoint-ids`. Bypasses the interactive picker. Mutually exclusive
    # with deploy_config_path.
    checkpoint_ids: List[str] = field(default_factory=list)


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
    model_id: str


class DeploySuccessResult(BaseModel):
    deploy_config: DeployCheckpointsConfigComplete
    truss_config: Optional[truss_config.TrussConfig]
    model_version: Optional[DeploySuccessModelVersion]
