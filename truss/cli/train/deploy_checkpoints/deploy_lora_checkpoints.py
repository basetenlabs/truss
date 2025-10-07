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
    return LoRACheckpoint(
        training_job_id=job_id,
        lora_details=LoRADetails(rank=_get_lora_rank(checkpoint)),
        checkpoint_name=checkpoint_id,
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
