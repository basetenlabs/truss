import os
from pathlib import Path

from jinja2 import Template

from truss.base import truss_config
from truss.cli.train.types import DeployCheckpointsConfigComplete
from truss_train.definitions import (
    ALLOWED_LORA_RANKS,
    DEFAULT_LORA_RANK,
    LoRACheckpoint,
    LoRADetails,
    SecretReference,
    TrainingArtifactReferencePathDetails,
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
    path_details = [
        TrainingArtifactReferencePathDetails(
            path_reference=f"rank-0/{checkpoint_id}/", recursive=True
        )
    ]
    return LoRACheckpoint(
        training_job_id=job_id,
        path_details=path_details,
        lora_details=LoRADetails(rank=_get_lora_rank(checkpoint)),
    )


def render_vllm_lora_truss_config(
    checkpoint_deploy: DeployCheckpointsConfigComplete,
) -> truss_config.TrussConfig:
    """Render truss config specifically for LoRA checkpoints using vLLM."""
    truss_deploy_config = truss_config.TrussConfig.from_yaml(
        Path(os.path.dirname(__file__), "..", "deploy_from_checkpoint_config.yml")
    )
    if not truss_deploy_config.docker_server:
        raise ValueError(
            "Unexpected checkpoint deployment config: missing docker_server"
        )

    truss_deploy_config.model_name = checkpoint_deploy.model_name
    truss_deploy_config.training_checkpoints = (
        checkpoint_deploy.checkpoint_details.to_truss_config()
    )
    truss_deploy_config.resources = checkpoint_deploy.compute.to_truss_config()
    for key, value in checkpoint_deploy.runtime.environment_variables.items():
        if isinstance(value, SecretReference):
            truss_deploy_config.secrets[value.name] = "set token in baseten workspace"
        else:
            truss_deploy_config.environment_variables[key] = value

    start_command_envvars = ""
    for key, value in checkpoint_deploy.runtime.environment_variables.items():
        if isinstance(value, SecretReference):
            truss_deploy_config.secrets[value.name] = "set token in baseten workspace"
            start_command_envvars = f"{key}=$(cat /secrets/{value.name})"

    checkpoint_parts = []
    for (
        truss_checkpoint
    ) in truss_deploy_config.training_checkpoints.artifact_references:  # type: ignore
        ckpt_path = Path(
            truss_deploy_config.training_checkpoints.download_folder,  # type: ignore
            truss_checkpoint.training_job_id,
            truss_checkpoint.path_details[0].path_reference,
        )
        checkpoint_parts.append(f"{truss_checkpoint.training_job_id}={ckpt_path}")
    checkpoint_str = " ".join(checkpoint_parts)
    max_lora_rank = max(
        [
            checkpoint.lora_details.rank or DEFAULT_LORA_RANK
            for checkpoint in checkpoint_deploy.checkpoint_details.checkpoints
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
    truss_deploy_config.docker_server.start_command = VLLM_LORA_START_COMMAND.render(
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
