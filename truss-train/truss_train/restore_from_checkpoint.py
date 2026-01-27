from truss_train import (
    BasetenCheckpoint,
    CheckpointingConfig,
    Image,
    LoadCheckpointConfig,
    Runtime,
    TrainingJob,
    TrainingProject,
)

load_checkpoint_config = LoadCheckpointConfig(enabled=True)

load_from_most_recent_checkpoint = BasetenCheckpoint.from_latest_checkpoint()

load_most_recent_checkpoint = BasetenCheckpoint.from_latest_checkpoint(
    job_id="lqz4pw4",  # Optional
    project_name="first-project",  # Optional
)

load_from_named_checkpoint = BasetenCheckpoint.from_named_checkpoint(
    checkpoint_name="checkpoint-24", job_id="lqz4pw4"
)

load_checkpoint_config = LoadCheckpointConfig(
    enabled=True,
    download_folder="/tmp/custom_location",  # default is None -> default path set by server-side
    checkpoints=[load_from_most_recent_checkpoint, load_from_named_checkpoint],
)

checkpointing_config = CheckpointingConfig(enabled=True)

job = TrainingJob(
    image=Image(base_image="ghcr.io/baseten-ai/truss-train-base:latest"),
    runtime=Runtime(
        checkpointing_config=checkpointing_config,
        load_checkpoint_config=load_checkpoint_config,
    ),
)

project = TrainingProject(name="new-project", job=job)
