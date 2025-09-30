from truss_train.definitions import WhisperCheckpoint


def hydrate_whisper_checkpoint(
    job_id: str, checkpoint_id: str, checkpoint: dict
) -> WhisperCheckpoint:
    """Create a Checkpoint object for whisper model weights."""
    # NOTE: Slash at the end is important since it means the checkpoint is a directory
    paths = [f"rank-0/{checkpoint_id}/"]
    return WhisperCheckpoint(training_job_id=job_id, paths=paths)
