from truss.base import truss_config
from truss_train.definitions import (
    CacheConfig,
    Compute,
    Image,
    InteractiveSession,
    InteractiveSessionProvider,
    InteractiveSessionTrigger,
    Runtime,
    TrainingJob,
    TrainingProject,
)

DEFAULT_BASE_IMAGE = "nvidia/cuda:12.8.1-devel-ubuntu24.04"


def build_workstation_project(
    accelerator: str,
    gpu_count: int,
    project_id: str,
    base_image: str = DEFAULT_BASE_IMAGE,
) -> TrainingProject:
    accel_enum = truss_config.Accelerator(accelerator)

    compute = Compute(
        node_count=1,
        accelerator=truss_config.AcceleratorSpec(
            accelerator=accel_enum, count=gpu_count
        ),
    )

    runtime = Runtime(
        start_commands=["sleep infinity"],
        cache_config=CacheConfig(enabled=True, require_cache_affinity=False),
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
