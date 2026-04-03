from truss_train.definitions import WhisperCheckpoint


def hydrate_whisper_checkpoint(
    job_id: str, checkpoint_id: str, checkpoint: dict
) -> WhisperCheckpoint:
    """Create a Checkpoint object for whisper model weights."""
    return WhisperCheckpoint(training_job_id=job_id, checkpoint_name=checkpoint_id)
