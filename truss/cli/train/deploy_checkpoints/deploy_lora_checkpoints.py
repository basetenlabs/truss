from pathlib import Path

from truss_train.definitions import (
    ALLOWED_LORA_RANKS,
    DEFAULT_LORA_RANK,
    LoRACheckpoint,
    LoRADetails,
)


def hydrate_lora_checkpoint(
    job_id: str, checkpoint_id: str, checkpoint: dict
) -> LoRACheckpoint:
    """Create a LoRA-specific Checkpoint object."""
    # NOTE: Slash at the end is important since it means the checkpoint is a directory
    paths = [f"rank-0/{checkpoint_id}/"]
    return LoRACheckpoint(
        training_job_id=job_id,
        paths=paths,
        lora_details=LoRADetails(rank=_get_lora_rank(checkpoint)),
    )


def _get_lora_rank(checkpoint_resp: dict) -> int:
    """Extract and validate LoRA rank from checkpoint response."""
    lora_adapter_config = checkpoint_resp.get("lora_adapter_config") or {}
    lora_rank = lora_adapter_config.get("r") or DEFAULT_LORA_RANK

    # If the API returns an invalid value, raise an error
    if lora_rank not in ALLOWED_LORA_RANKS:
        raise ValueError(
            f"LoRA rank {lora_rank} from checkpoint is not in allowed values {sorted(ALLOWED_LORA_RANKS)}. "
            f"Please use a valid LoRA rank."
        )

    return lora_rank


def _build_lora_checkpoint_string(truss_deploy_config) -> str:
    """Build the checkpoint string for LoRA modules from truss deploy config."""
    checkpoint_parts = []
    for (
        truss_checkpoint
    ) in truss_deploy_config.training_checkpoints.artifact_references:  # type: ignore
        ckpt_path = Path(
            truss_deploy_config.training_checkpoints.download_folder,  # type: ignore
            truss_checkpoint.training_job_id,
            truss_checkpoint.paths[0],
        )
        checkpoint_parts.append(f"{truss_checkpoint.training_job_id}={ckpt_path}")

    return " ".join(checkpoint_parts)
