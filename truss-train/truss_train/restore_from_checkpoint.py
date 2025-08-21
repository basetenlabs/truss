import truss_train

# Default - restores checkpoints from the previous checkpointing job in the project
# Restores all contents
restore_config = truss_train.RestoreFromCheckpointConfig(enabled=True)

# Restore only the latest chcekpoint
restore_config = truss_train.RestoreFromCheckpointConfig(
    enabled=True, restore=truss_train.RestoreCheckpoint(most_recent_checkpoint=True)
)

# Restore so that the contents are stored in a specific subdirectory
restore_config = truss_train.RestoreFromCheckpointConfig(
    enabled=True,
    restore=truss_train.RestoreCheckpoint(most_recent_checkpoint=True),
    mount_subdir="previous_ckpts",
)

# Restore some specific paths
restore_config = truss_train.RestoreFromCheckpointConfig(
    enabled=True,
    restore=truss_train.RestorePaths(paths=["checkpoint-24/*", "tokenizer.json"]),
)

# Restore from a specific job in a specific project
restore_config = truss_train.RestoreFromCheckpointConfig(
    enabled=True, job_id="1234567890", project_name="test-project"
)

checkpointing_config = truss_train.CheckpointingConfig(
    enabled=True, restore_config=restore_config
)

job = truss_train.TrainingJob(
    image=truss_train.Image(base_image="ghcr.io/baseten-ai/truss-train-base:latest"),
    runtime=truss_train.Runtime(checkpointing_config=checkpointing_config),
)

project = truss_train.TrainingProject(name="new-project", job=job)
