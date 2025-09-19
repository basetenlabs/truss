import os
from pathlib import Path
from typing import Any, Dict

from truss.base import truss_config
from truss.cli.train.types import DeployCheckpointsConfigComplete
from truss_train.definitions import ModelWeightsFormat, SecretReference

START_COMMAND_ENVVAR_NAME = "BT_DOCKER_SERVER_START_CMD"


def setup_base_truss_config(
    checkpoint_deploy: DeployCheckpointsConfigComplete,
) -> truss_config.TrussConfig:
    """Set up the base truss config with common properties."""
    truss_deploy_config = None
    truss_base_file = (
        "deploy_from_checkpoint_config_whisper.yml"
        if checkpoint_deploy.model_weight_format == ModelWeightsFormat.WHISPER
        else "deploy_from_checkpoint_config.yml"
    )
    truss_deploy_config = truss_config.TrussConfig.from_yaml(
        Path(os.path.dirname(__file__), "..", truss_base_file)
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

    return truss_deploy_config


def setup_environment_variables_and_secrets(
    truss_deploy_config: truss_config.TrussConfig,
    checkpoint_deploy: DeployCheckpointsConfigComplete,
) -> str:
    """Set up environment variables and secrets, return start command envvars string."""
    start_command_envvars = ""

    for key, value in checkpoint_deploy.runtime.environment_variables.items():
        if isinstance(value, SecretReference):
            truss_deploy_config.secrets[value.name] = "set token in baseten workspace"
            start_command_envvars = f"{key}=$(cat /secrets/{value.name})"
        else:
            truss_deploy_config.environment_variables[key] = value

    return start_command_envvars


def prepare_graphql_deployment_request(
    checkpoint_deploy: DeployCheckpointsConfigComplete,
    oracle_name: str,
    instance_type_id: str,
) -> Dict[str, Any]:
    """
    Convert DeployCheckpointsConfigComplete to GraphQL request format.

    Args:
        checkpoint_deploy: The complete checkpoint deployment configuration
        oracle_name: Name for the oracle/deployment
        instance_type_id: The instance type ID for deployment

    Returns:
        Dictionary in the format expected by the GraphQL mutation
    """
    # Prepare weights sources from checkpoints
    weights_sources = []
    for checkpoint in checkpoint_deploy.checkpoint_details.checkpoints:
        weight_source = {
            "weight_source_type": "B10_CHECKPOINTING",
            "b10_training_checkpoint_weights_source": {
                "checkpoint": {
                    "training_job_id": checkpoint.training_job_id,
                    "checkpoint_name": checkpoint.paths[0].strip("/").split("/")[-1],
                }
            },
        }
        weights_sources.append(weight_source)

    # Prepare environment variables
    env_vars = []
    for key, value in checkpoint_deploy.runtime.environment_variables.items():
        if isinstance(value, SecretReference):
            # For secret references, we'll need to handle this differently
            # For now, we'll pass the secret name as the value
            env_vars.append(
                {"name": key, "value": value.name, "is_secret_reference": True}
            )
        else:
            env_vars.append(
                {"name": key, "value": str(value), "is_secret_reference": False}
            )

    # Determine stack type based on model weight format
    stack_type = "VLLM"  # Default to VLLM

    # Prepare inference stack
    inference_stack = {"stack_type": stack_type, "environment_variables": env_vars}

    # Build the complete request
    request_data = {
        "request": {
            "metadata": {"oracle_name": oracle_name},
            "weights_sources": weights_sources,
            "inference_stack": inference_stack,
            "instance_type_id": instance_type_id,
        }
    }

    return request_data
