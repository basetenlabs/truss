from pathlib import Path

from truss_train.definitions import FullCheckpoint


def hydrate_full_checkpoint(
    job_id: str, checkpoint_id: str, checkpoint: dict
) -> FullCheckpoint:
    """Create a Checkpoint object for full model weights."""
    # NOTE: Slash at the end is important since it means the checkpoint is a directory
    paths = [f"rank-0/{checkpoint_id}/"]
    return FullCheckpoint(training_job_id=job_id, paths=paths)


def build_full_checkpoint_string(truss_deploy_config) -> str:
    """Build checkpoint string from artifact references for full checkpoints.

    Args:
        truss_deploy_config: The truss deploy configuration containing training checkpoints.

    Returns:
        A space-separated string of checkpoint paths.
    """
    checkpoint_parts = []
    for (
        truss_checkpoint
    ) in truss_deploy_config.training_checkpoints.artifact_references:  # type: ignore
        ckpt_path = Path(
            truss_deploy_config.training_checkpoints.download_folder,  # type: ignore
            truss_checkpoint.training_job_id,
            truss_checkpoint.paths[0],
        )
        checkpoint_parts.append(str(ckpt_path))

    return " ".join(checkpoint_parts)
