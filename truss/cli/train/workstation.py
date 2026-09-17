import shutil
from pathlib import Path
from typing import Optional

from truss.base import truss_config
from truss.base.constants import WORKSTATION_TEMPLATE_DIR
from truss.base.truss_config import Accelerator
from truss_train.definitions import (
    BasetenCheckpoint,
    CacheConfig,
    CheckpointingConfig,
    Compute,
    Image,
    InteractiveSession,
    InteractiveSessionProvider,
    InteractiveSessionTrigger,
    LoadCheckpointConfig,
    Runtime,
    TrainingJob,
    TrainingProject,
)

DEFAULT_BASE_IMAGE = "nvidia/cuda:12.9.1-devel-ubuntu24.04"
B300_BASE_IMAGE = "nvidia/cuda:13.0.3-devel-ubuntu24.04"
SUPPORTED_WORKSTATION_ACCELERATORS = [
    acc.value for acc in Accelerator if acc.value != Accelerator._B10.value
]


def default_base_image(accelerator: str) -> str:
    if accelerator in (Accelerator.B300.value, Accelerator.GB300.value):
        return B300_BASE_IMAGE
    return DEFAULT_BASE_IMAGE


def copy_workstation_templates(target_dir: Path) -> None:
    """Copy workstation SLURM setup scripts to the target directory."""
    for script in WORKSTATION_TEMPLATE_DIR.iterdir():
        if script.is_file() and script.suffix == ".sh":
            dest = target_dir / script.name
            shutil.copy2(str(script), str(dest))
            dest.chmod(0o755)


ORCHESTRATOR_START_COMMANDS = {"slurm": "bash /b10/workspace/setup_slurm.sh"}

SSH_HOSTNAME_SUFFIX = "ssh.baseten.co"
# Remote name that `truss` (and therefore ~/.trussrc) uses by default.
DEFAULT_SSH_REMOTE = "baseten"


def workstation_ssh_hostnames(
    job_id: str, node_count: int, remote: Optional[str] = None
) -> list[str]:
    """SSH hostnames for a workstation's nodes, ordered by node rank.

    The remote segment is optional in the hostname grammar that
    `truss.cli.proxy_command` parses. It is only needed to disambiguate when
    ~/.trussrc holds several remotes, so it is omitted for the default remote to
    keep the common case short.
    """
    host_suffix = SSH_HOSTNAME_SUFFIX
    if remote and remote != DEFAULT_SSH_REMOTE:
        host_suffix = f"{remote}.{SSH_HOSTNAME_SUFFIX}"
    return [f"training-job-{job_id}-{rank}.{host_suffix}" for rank in range(node_count)]


def build_workstation_project(
    accelerator: str,
    gpu_count: int,
    project_id: str,
    base_image: Optional[str] = None,
    node_count: int = 1,
    orchestrator: str = "slurm",
    enable_checkpointing: bool = False,
    checkpoint_path: Optional[str] = None,
    checkpoint_volume_size: Optional[int] = None,
    checkpoint_from_job: Optional[str] = None,
) -> TrainingProject:
    accel_enum = truss_config.Accelerator(accelerator)

    compute = Compute(
        node_count=node_count,
        accelerator=truss_config.AcceleratorSpec(
            accelerator=accel_enum, count=gpu_count
        ),
    )

    load_checkpoint_config = None
    if checkpoint_from_job:
        load_checkpoint_config = LoadCheckpointConfig(
            enabled=True,
            checkpoints=[
                BasetenCheckpoint.from_latest_checkpoint(job_id=checkpoint_from_job)
            ],
        )

    if node_count > 1:
        start_commands = [ORCHESTRATOR_START_COMMANDS[orchestrator]]
    else:
        start_commands = ["sleep infinity"]

    runtime = Runtime(
        start_commands=start_commands,
        cache_config=CacheConfig(enabled=True, require_cache_affinity=False),
        checkpointing_config=CheckpointingConfig(
            enabled=enable_checkpointing,
            checkpoint_path=checkpoint_path,
            volume_size_gib=checkpoint_volume_size,
        ),
        load_checkpoint_config=load_checkpoint_config,
    )

    interactive_session = InteractiveSession(
        trigger=InteractiveSessionTrigger.ON_STARTUP,
        session_provider=InteractiveSessionProvider.SSH,
    )

    job = TrainingJob(
        image=Image(base_image=base_image or default_base_image(accelerator)),
        compute=compute,
        runtime=runtime,
        interactive_session=interactive_session,
    )

    return TrainingProject(name=project_id, job=job)
