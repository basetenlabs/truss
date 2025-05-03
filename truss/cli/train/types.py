from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from truss_train.definitions import CheckpointDeployConfig


@dataclass
class PrepareCheckpointArgs:
    project_id: Optional[str]
    job_id: Optional[str]
    deploy_config_path: Optional[str]


@dataclass
class PrepareCheckpointResult:
    truss_directory: Path
    checkpoint_deploy_config: CheckpointDeployConfig
