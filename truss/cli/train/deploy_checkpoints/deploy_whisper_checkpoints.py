from jinja2 import Template

from truss.base import truss_config
from truss.cli.train.deploy_checkpoints.deploy_checkpoints_helpers import (
    START_COMMAND_ENVVAR_NAME,
)
from truss.cli.train.deploy_checkpoints.deploy_full_checkpoints import (
    build_full_checkpoint_string,
)
from truss.cli.train.types import DeployCheckpointsConfigComplete
from truss_train.definitions import WhisperCheckpoint

from .deploy_checkpoints_helpers import (
    setup_base_truss_config,
    setup_environment_variables_and_secrets,
)

VLLM_WHISPER_START_COMMAND = Template(
    "sh -c '{% if envvars %}{{ envvars }} {% endif %}"
    'HF_TOKEN="$$(cat /secrets/hf_access_token)" && export HF_TOKEN && '
    "vllm serve {{ model_path }} --port 8000 --tensor-parallel-size {{ specify_tensor_parallelism }}'"
)


def render_vllm_whisper_truss_config(
    checkpoint_deploy: DeployCheckpointsConfigComplete,
) -> truss_config.TrussConfig:
    """Render truss config specifically for whisper checkpoints using vLLM."""
    truss_deploy_config = setup_base_truss_config(checkpoint_deploy)

    start_command_envvars = setup_environment_variables_and_secrets(
        truss_deploy_config, checkpoint_deploy
    )

    checkpoint_str = build_full_checkpoint_string(truss_deploy_config)

    accelerator = checkpoint_deploy.compute.accelerator

    start_command_args = {
        "model_path": checkpoint_str,
        "envvars": start_command_envvars,
        "specify_tensor_parallelism": accelerator.count if accelerator else 1,
    }
    # Note: we set the start command as an environment variable in supervisord config.
    # This is so that we don't have to change the supervisord config when the start command changes.
    # Our goal is to reduce the number of times we need to rebuild the image, and allow us to deploy faster.
    start_command = VLLM_WHISPER_START_COMMAND.render(**start_command_args)
    truss_deploy_config.environment_variables[START_COMMAND_ENVVAR_NAME] = start_command
    # Note: supervisord uses the convention %(ENV_VAR_NAME)s to access environment variable VAR_NAME
    truss_deploy_config.docker_server.start_command = (  # type: ignore[union-attr]
        f"%(ENV_{START_COMMAND_ENVVAR_NAME})s"
    )

    return truss_deploy_config


def hydrate_whisper_checkpoint(
    job_id: str, checkpoint_id: str, checkpoint: dict
) -> WhisperCheckpoint:
    """Create a Checkpoint object for whisper model weights."""
    # NOTE: Slash at the end is important since it means the checkpoint is a directory
    paths = [f"rank-0/{checkpoint_id}/"]
    return WhisperCheckpoint(training_job_id=job_id, paths=paths)
