from jinja2 import Template

from truss.base import truss_config
from truss.cli.train.types import DeployCheckpointsConfigComplete
from truss_train.definitions import FullCheckpoint

from .deploy_checkpoints_helpers import (
    build_checkpoint_string,
    setup_base_truss_config,
    setup_environment_variables_and_secrets,
)

VLLM_FULL_START_COMMAND = Template(
    'sh -c "{%if envvars %}{{ envvars }} {% endif %}vllm serve {{ model_path }}'
    + " --port 8000"
    + " --tensor-parallel-size {{ specify_tensor_parallelism }}"
    + " --dtype bfloat16"
    + '"'
)


def render_vllm_full_truss_config(
    checkpoint_deploy: DeployCheckpointsConfigComplete,
) -> truss_config.TrussConfig:
    """Render truss config specifically for full checkpoints using vLLM."""
    truss_deploy_config = setup_base_truss_config(checkpoint_deploy)

    start_command_envvars = setup_environment_variables_and_secrets(
        truss_deploy_config, checkpoint_deploy
    )

    checkpoint_str = build_checkpoint_string(truss_deploy_config)

    accelerator = checkpoint_deploy.compute.accelerator

    start_command_args = {
        "model_path": checkpoint_str,
        "envvars": start_command_envvars,
        "specify_tensor_parallelism": accelerator.count,
    }
    truss_deploy_config.docker_server.start_command = VLLM_FULL_START_COMMAND.render(  # type: ignore[union-attr]
        **start_command_args
    )

    return truss_deploy_config


def hydrate_full_checkpoint(
    job_id: str, checkpoint_id: str, checkpoint: dict
) -> FullCheckpoint:
    """Create a LoRA-specific Checkpoint object."""
    # NOTE: Slash at the end is important since it means the checkpoint is a directory
    paths = [f"rank-0/{checkpoint_id}/"]
    return FullCheckpoint(training_job_id=job_id, paths=paths)
