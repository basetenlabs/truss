from jinja2 import Template

from truss.base import truss_config
from truss.cli.train.types import DeployCheckpointsConfigComplete
from truss_train.definitions import (
    ALLOWED_LORA_RANKS,
    DEFAULT_LORA_RANK,
    LoRACheckpoint,
    LoRADetails,
)

from .deploy_checkpoints_helpers import (
    build_checkpoint_string,
    setup_base_truss_config,
    setup_environment_variables_and_secrets,
)

VLLM_LORA_START_COMMAND = Template(
    'sh -c "{%if envvars %}{{ envvars }} {% endif %}vllm serve {{ base_model_id }}'
    + " --port 8000"
    + "{{ specify_tensor_parallelism }}"
    + " --enable-lora"
    + " --max-lora-rank {{ max_lora_rank }}"
    + " --dtype bfloat16"
    + ' --lora-modules {{ lora_modules }}"'
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


def render_vllm_lora_truss_config(
    checkpoint_deploy: DeployCheckpointsConfigComplete,
) -> truss_config.TrussConfig:
    """Render truss config specifically for LoRA checkpoints using vLLM."""
    truss_deploy_config = setup_base_truss_config(checkpoint_deploy)

    start_command_envvars = setup_environment_variables_and_secrets(
        truss_deploy_config, checkpoint_deploy
    )

    checkpoint_str = build_checkpoint_string(truss_deploy_config)
    max_lora_rank = max(
        [
            checkpoint.lora_details.rank or DEFAULT_LORA_RANK
            for checkpoint in checkpoint_deploy.checkpoint_details.checkpoints
            if hasattr(checkpoint, "lora_details") and checkpoint.lora_details
        ]
    )
    accelerator = checkpoint_deploy.compute.accelerator
    if accelerator:
        specify_tensor_parallelism = f" --tensor-parallel-size {accelerator.count}"
    else:
        specify_tensor_parallelism = ""

    start_command_args = {
        "base_model_id": checkpoint_deploy.checkpoint_details.base_model_id,
        "lora_modules": checkpoint_str,
        "envvars": start_command_envvars,
        "max_lora_rank": max_lora_rank,
        "specify_tensor_parallelism": specify_tensor_parallelism,
    }
    truss_deploy_config.docker_server.start_command = VLLM_LORA_START_COMMAND.render(  # type: ignore[union-attr]
        **start_command_args
    )
    return truss_deploy_config


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
