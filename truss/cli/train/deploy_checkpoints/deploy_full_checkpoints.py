from pathlib import Path

from jinja2 import Template

from truss.base import truss_config
from truss.cli.train.deploy_checkpoints.deploy_checkpoints_helpers import (
    START_COMMAND_ENVVAR_NAME,
)
from truss.cli.train.types import DeployCheckpointsConfigComplete
from truss_train.definitions import FullCheckpoint

from .deploy_checkpoints_helpers import (
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

    checkpoint_str = _build_full_checkpoint_string(truss_deploy_config)

    accelerator = checkpoint_deploy.compute.accelerator

    start_command_args = {
        "model_path": checkpoint_str,
        "envvars": start_command_envvars,
        "specify_tensor_parallelism": accelerator.count if accelerator else 1,
    }
    # Note: we set the start command as an environment variable in supervisord config.
    # This is so that we don't have to change the supervisord config when the start command changes.
    # Our goal is to reduce the number of times we need to rebuild the image, and allow us to deploy faster.
    start_command = VLLM_FULL_START_COMMAND.render(**start_command_args)
    truss_deploy_config.environment_variables[START_COMMAND_ENVVAR_NAME] = start_command
    # Note: supervisord uses the convention %(ENV_VAR_NAME)s to access environment variable VAR_NAME
    truss_deploy_config.docker_server.start_command = (  # type: ignore[union-attr]
        f"%(ENV_{START_COMMAND_ENVVAR_NAME})s"
    )

    return truss_deploy_config


def hydrate_full_checkpoint(
    job_id: str, checkpoint_id: str, checkpoint: dict
) -> FullCheckpoint:
    """Create a Checkpoint object for full model weights."""
    # NOTE: Slash at the end is important since it means the checkpoint is a directory
    paths = [f"rank-0/{checkpoint_id}/"]
    return FullCheckpoint(training_job_id=job_id, paths=paths)


def _build_full_checkpoint_string(truss_deploy_config) -> str:
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
