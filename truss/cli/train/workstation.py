import shutil
from pathlib import Path
from typing import Optional

from truss.base import truss_config
from truss.base.constants import WORKSTATION_TEMPLATE_DIR
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

DEFAULT_BASE_IMAGE = "nvidia/cuda:12.8.1-devel-ubuntu24.04"


def copy_workstation_templates(target_dir: Path) -> None:
    """Copy workstation SLURM setup scripts to the target directory."""
    for script in WORKSTATION_TEMPLATE_DIR.iterdir():
        if script.is_file() and script.suffix == ".sh":
            dest = target_dir / script.name
            shutil.copy2(str(script), str(dest))
            dest.chmod(0o755)


def build_workstation_project(
    accelerator: str,
    gpu_count: int,
    project_id: str,
    base_image: str = DEFAULT_BASE_IMAGE,
    node_count: int = 1,
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
        start_commands = ["bash /b10/workspace/setup_slurm.sh"]
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
        image=Image(base_image=base_image),
        compute=compute,
        runtime=runtime,
        interactive_session=interactive_session,
    )

    return TrainingProject(name=project_id, job=job)
