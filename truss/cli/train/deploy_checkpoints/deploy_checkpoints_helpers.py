import os
from pathlib import Path
<<<<<<< Updated upstream
=======
from typing import Any, Dict, Optional
>>>>>>> Stashed changes

from truss.base import truss_config
from truss.cli.train.types import DeployCheckpointsConfigComplete
from truss.remote.baseten.remote import BasetenRemote
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
<<<<<<< Updated upstream
=======


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


def get_instance_types_map(remote_provider: BasetenRemote) -> Dict[str, Dict[str, Any]]:
    """
    Fetch all available instance types and create a map of instance type ID to details.

    Args:
        remote_provider: The Baseten remote provider

    Returns:
        Dictionary mapping instance type ID to instance type details
    """
    try:
        instance_types = remote_provider.api.list_instance_types()
        return {instance_type["id"]: instance_type for instance_type in instance_types}
    except Exception:
        # If we can't fetch instance types, return empty dict
        # This allows the deployment to proceed with user-provided instance type
        return {}


def validate_instance_type_id(
    instance_type_id: str, remote_provider: BasetenRemote
) -> bool:
    """
    Validate that the provided instance type ID exists in the available instance types.

    Args:
        instance_type_id: The instance type ID to validate
        remote_provider: The Baseten remote provider

    Returns:
        True if the instance type ID is valid, False otherwise
    """
    instance_types_map = get_instance_types_map(remote_provider)
    return instance_type_id in instance_types_map


def get_instance_type_details(
    instance_type_id: str, remote_provider: BasetenRemote
) -> Optional[Dict[str, Any]]:
    """
    Get details for a specific instance type ID.

    Args:
        instance_type_id: The instance type ID to get details for
        remote_provider: The Baseten remote provider

    Returns:
        Dictionary containing instance type details, or None if not found
    """
    instance_types_map = get_instance_types_map(remote_provider)
    return instance_types_map.get(instance_type_id)


def infer_instance_type_id(
    compute_config: Any, remote_provider: BasetenRemote
) -> Optional[str]:
    """
    Infer the appropriate instance type ID based on compute configuration.

    Args:
        compute_config: The compute configuration from DeployCheckpointsConfigComplete
        remote_provider: The Baseten remote provider

    Returns:
        The inferred instance type ID, or None if no match found
    """
    instance_types_map = get_instance_types_map(remote_provider)

    if not instance_types_map:
        return None

    # Extract compute requirements
    cpu_count = getattr(compute_config, "cpu_count", 0)
    memory = getattr(compute_config, "memory", "0Mi")
    accelerator = getattr(compute_config, "accelerator", None)

    # Convert memory to a comparable format (assume GiB)
    memory_gb = _parse_memory_to_gb(memory)

    # Find matching instance types
    candidates = []

    for instance_id, instance_details in instance_types_map.items():
        instance_cpu = instance_details.get("cpu_count", 0)
        instance_memory_gb = _parse_memory_to_gb(instance_details.get("memory", "0Mi"))
        instance_gpu_type = instance_details.get("gpu_type")
        instance_gpu_count = instance_details.get("gpu_count", 0)

        # Check CPU match
        cpu_match = cpu_count == 0 or instance_cpu >= cpu_count

        # Check memory match
        memory_match = memory_gb == 0 or instance_memory_gb >= memory_gb

        # Check accelerator match
        accelerator_match = True
        if accelerator:
            if accelerator.accelerator and instance_gpu_type:
                # Map truss accelerator types to instance type GPU types
                accelerator_match = _matches_gpu_type(
                    accelerator.accelerator, instance_gpu_type
                )
            if accelerator.count and instance_gpu_count:
                accelerator_match = (
                    accelerator_match and instance_gpu_count >= accelerator.count
                )

        if cpu_match and memory_match and accelerator_match:
            candidates.append((instance_id, instance_details))

    if not candidates:
        return None

    # Sort by preference (prefer smaller instances that meet requirements)
    candidates.sort(key=lambda x: (x[1].get("cpu_count", 0), x[1].get("memory", "0Mi")))

    return candidates[0][0]


def _parse_memory_to_gb(memory_str: str) -> float:
    """
    Parse memory string to GB.

    Args:
        memory_str: Memory string like "16Gi", "32GB", "8Mi"

    Returns:
        Memory in GB as float
    """
    if not memory_str:
        return 0.0

    memory_str = memory_str.upper().strip()

    # Extract number and unit
    import re

    match = re.match(r"(\d+(?:\.\d+)?)\s*(GI?B?|MI?B?|GB?|MB?)", memory_str)
    if not match:
        return 0.0

    value = float(match.group(1))
    unit = match.group(2)

    # Convert to GB
    if unit.startswith("M"):
        return value / 1024  # MB to GB
    elif unit.startswith("G"):
        return value  # Already GB
    else:
        return 0.0


def _matches_gpu_type(truss_accelerator: str, instance_gpu_type: str) -> bool:
    """
    Check if truss accelerator type matches instance GPU type.

    Args:
        truss_accelerator: Truss accelerator type (e.g., "A10G", "T4")
        instance_gpu_type: Instance GPU type from API

    Returns:
        True if they match, False otherwise
    """
    if not truss_accelerator or not instance_gpu_type:
        return False

    # Normalize both types for comparison
    truss_type = truss_accelerator.upper().strip()
    instance_type = instance_gpu_type.upper().strip()

    # Direct match
    if truss_type == instance_type:
        return True

    # Handle common variations
    if truss_type in instance_type or instance_type in truss_type:
        return True

    return False
>>>>>>> Stashed changes
