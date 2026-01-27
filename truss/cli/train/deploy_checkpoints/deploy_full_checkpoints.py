from truss_train.definitions import FullCheckpoint


def hydrate_full_checkpoint(
    job_id: str, checkpoint_id: str, checkpoint: dict
) -> FullCheckpoint:
    """Create a Checkpoint object for full model weights."""
    # NOTE: Slash at the end is important since it means the checkpoint is a directory
    return FullCheckpoint(training_job_id=job_id, checkpoint_name=checkpoint_id)
