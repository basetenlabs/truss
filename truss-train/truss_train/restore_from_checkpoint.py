import truss_train

# Default - restores checkpoints from the previous checkpointing job in the project
# Restores all contents
restore_config = truss_train.RestoreFromCheckpointConfig(enabled=True)

# This is the one I like the most since it takes no arguments and it theory it could always be on - even if no checkpointing is happening.
# So if a job ever fails for an unknown reason, you could just do truss train recreate and it will restore from the most recent checkpoint (no code changes needed)
restore_from_most_recent_checkpoint = truss_train.RestoreMostRecentCheckpoint()

restore_from_named_checkpoint = truss_train.RestoreNamedCheckpoint(
    checkpoint_name="checkpoint-24"
)
restore_from_paths = truss_train.RestorePaths(paths=["rank-0/checkpoint-10/"])

# Restore only the latest chcekpoint
restore_config = truss_train.RestoreFromCheckpointConfig(
    enabled=True,
    restore=restore_from_most_recent_checkpoint,  # Or restore_from_named_checkpoint or restore_from_paths
    mount_subdir="/tmp/custom_location",  # default is /tmp/restored_checkpoints
    # Could also pass in job_id and project_name
)

checkpointing_config = truss_train.CheckpointingConfig(
    enabled=True, restore_config=restore_config
)

job = truss_train.TrainingJob(
    image=truss_train.Image(base_image="ghcr.io/baseten-ai/truss-train-base:latest"),
    runtime=truss_train.Runtime(checkpointing_config=checkpointing_config),
)

project = truss_train.TrainingProject(name="new-project", job=job)
