import json
import re
from collections import OrderedDict
from typing import List, Optional, Union

import rich_click as click
from InquirerPy import inquirer

from truss.base import truss_config
from truss.cli.train import common
from truss.cli.train.types import (
    DeployCheckpointsConfigComplete,
    DeploySuccessModelVersion,
    DeploySuccessResult,
)
from truss.cli.utils.output import console
from truss.remote.baseten.remote import BasetenRemote
from truss_train.definitions import (
    Checkpoint,
    CheckpointList,
    Compute,
    DeployCheckpointsConfig,
    DeployCheckpointsRuntime,
    ModelWeightsFormat,
    SecretReference,
)

from .deploy_full_checkpoints import hydrate_full_checkpoint
from .deploy_lora_checkpoints import hydrate_lora_checkpoint
from .deploy_whisper_checkpoints import hydrate_whisper_checkpoint

HF_TOKEN_ENVVAR_NAME = "HF_TOKEN"
# If we change this, make sure to update the logic in backend codebase
CHECKPOINT_PATTERN = re.compile(r".*checkpoint-\d+(?:-\d+)?$")
ALLOWED_DEPLOYMENT_NAMES = re.compile(r"^[0-9a-zA-Z_\-\.]*$")


def create_model_version_from_inference_template(
    remote_provider: BasetenRemote,
    checkpoint_deploy_config: DeployCheckpointsConfig,
    project_id: Optional[str],
    job_id: Optional[str],
    dry_run: bool,
) -> DeploySuccessResult:
    checkpoint_deploy_config = _hydrate_deploy_config(
        checkpoint_deploy_config, remote_provider, project_id, job_id
    )

    request_data = _build_inference_template_request(
        checkpoint_deploy_config=checkpoint_deploy_config,
        remote_provider=remote_provider,
        dry_run=dry_run,
    )

    # Call the GraphQL mutation to create model version from inference template
    try:
        result = remote_provider.api.create_model_version_from_inference_template(
            request_data
        )
        truss_config_result = _get_truss_config_from_result(result)

        model_version = None
        if result and result.get("model_version"):
            console.print(
                f"Successfully created model version: {result['model_version']['name']}",
                style="green",
            )
            console.print(
                f"Model version ID: {result['model_version']['id']}", style="yellow"
            )
            model_version = DeploySuccessModelVersion.model_validate(
                result["model_version"]
            )
        elif not dry_run:
            console.print(
                "Warning: Unexpected response format from server", style="yellow"
            )
            console.print(f"Response: {result}", style="yellow")

    except Exception as e:
        console.print(f"Error creating model version: {e}", style="red")
        raise

    return DeploySuccessResult(
        deploy_config=checkpoint_deploy_config,
        truss_config=truss_config_result,
        model_version=model_version,
    )


def _get_truss_config_from_result(result: dict) -> Optional[truss_config.TrussConfig]:
    if result and result.get("truss_config"):
        truss_config_dict = json.loads(result["truss_config"])
        return truss_config.TrussConfig.from_dict(truss_config_dict)
    # Although this should never happen, we defensively allow ourselves to return None
    # because we need a failure to handle the truss config doesn't necessarily mean we failed to deploy
    # the model version.
    console.print(
        "No truss config returned. Reach out to Baseten for support if this persists.",
        style="red",
    )
    return None


def _build_inference_template_request(
    checkpoint_deploy_config: DeployCheckpointsConfigComplete,
    remote_provider: BasetenRemote,
    dry_run: bool,
) -> dict:
    """
    Build the GraphQL request data structure for createModelVersionFromInferenceTemplate mutation.
    """

    # Build weights sources
    weights_sources = []
    for checkpoint in checkpoint_deploy_config.checkpoint_details.checkpoints:
        # Extract checkpoint name from the first path
        weights_source = {
            "weight_source_type": "B10_CHECKPOINTING",
            "b10_training_checkpoint_weights_source": {
                "checkpoint": {
                    "training_job_id": checkpoint.training_job_id,
                    "checkpoint_name": checkpoint.checkpoint_name,
                }
            },
        }
        weights_sources.append(weights_source)

    # Build environment variables
    environment_variables = []
    for name, value in checkpoint_deploy_config.runtime.environment_variables.items():
        if isinstance(value, SecretReference):
            env_var = {"name": name, "value": value.name, "is_secret_reference": True}
        else:
            env_var = {"name": name, "value": str(value), "is_secret_reference": False}
        environment_variables.append(env_var)

    # Build inference stack
    inference_stack = {
        "stack_type": "VLLM",
        "environment_variables": environment_variables,
    }

    # Get instance type ID from compute spec
    instance_type_id = _get_instance_type_id(
        checkpoint_deploy_config.compute, remote_provider
    )

    # Build the complete request
    request_data = {
        "metadata": {"oracle_name": checkpoint_deploy_config.model_name},
        "weights_sources": weights_sources,
        "inference_stack": inference_stack,
        "instance_type_id": instance_type_id,
        "dry_run": dry_run,
    }

    return request_data


def _get_instance_type_id(compute: Compute, remote_provider: BasetenRemote) -> str:
    """
    Get the instance type ID based on the compute specification.
    Fetches available instance types from the API and maps compute specs to instance type IDs.
    Only considers single-node instances (node_count == 1).
    """
    # step 1: fetch the instance types from the API
    instance_types = remote_provider.api.get_instance_types()
    # step 2: sort them into two different dictionaries, excluding multi-node instances:
    cpu_instance_types = {
        it.id: it for it in instance_types if it.gpu_count == 0 and it.node_count == 1
    }
    gpu_instance_types = {
        it.id: it for it in instance_types if it.gpu_count > 0 and it.node_count == 1
    }
    # step 3: if compute is cpu, find the smallest such cpu that matches the compute request
    if not compute.accelerator or compute.accelerator.accelerator is None:
        compute_as_truss_config = compute.to_truss_config()
        smallest_cpu_instance_type = None
        for it in cpu_instance_types.values():
            if (
                it.millicpu_limit / 1000 >= compute.cpu_count
                and it.memory_limit >= compute_as_truss_config.memory_in_bytes
            ):
                if (
                    smallest_cpu_instance_type is None
                    or it.millicpu_limit < smallest_cpu_instance_type.millicpu_limit
                ):
                    smallest_cpu_instance_type = it
        if not smallest_cpu_instance_type:
            raise ValueError(
                f"Unable to find single-node instance type for {compute.cpu_count} CPU and {compute.memory} memory. Reach out to Baseten for support if this persists."
            )
        return smallest_cpu_instance_type.id
    # step 4: if compute is gpu, find the smallest such gpu by instance type
    else:
        assert compute.accelerator.accelerator is not None
        compute_as_truss_config = compute.to_truss_config()
        smallest_gpu_instance_type = None
        for it in gpu_instance_types.values():
            if (
                it.gpu_type == compute.accelerator.accelerator.value
                and it.gpu_count >= compute.accelerator.count
            ):
                if (
                    smallest_gpu_instance_type is None
                    or it.gpu_count < smallest_gpu_instance_type.gpu_count
                ):
                    smallest_gpu_instance_type = it
        if not smallest_gpu_instance_type:
            raise ValueError(
                f"Unable to find single-node instance type for {compute.accelerator}:{compute.accelerator.count}. Reach out to Baseten for support if this persists."
            )
        return smallest_gpu_instance_type.id


def _validate_base_model_id(
    base_model_id: Optional[str], model_weight_format: ModelWeightsFormat
) -> None:
    """
    Validate that base model ID is present when required by the model weight format.
    """
    if not base_model_id and model_weight_format == ModelWeightsFormat.LORA:
        raise ValueError(
            "Unable to infer base model id. Reach out to Baseten for support."
        )


def _get_model_name(
    model_weight_format: ModelWeightsFormat, base_model_id: Optional[str]
) -> str:
    """
    Generate a model name based on the model weight format and base model ID.
    NOTE: Note not all checkpoints have a base model id nor need one
    """
    _validate_base_model_id(base_model_id, model_weight_format)
    default = (
        f"{base_model_id.split('/')[-1]}"  # type: ignore[union-attr]
        if model_weight_format == ModelWeightsFormat.LORA
        else ""
    )

    return inquirer.text(
        message=f"Enter the model name for your {model_weight_format.value} model.",
        validate=lambda s: s and s.strip(),
        default=default,
    ).execute()


def _hydrate_deploy_config(
    deploy_config: DeployCheckpointsConfig,
    remote_provider: BasetenRemote,
    project_id: Optional[str],
    job_id: Optional[str],
) -> DeployCheckpointsConfigComplete:
    checkpoint_details = _ensure_checkpoint_details(
        remote_provider, deploy_config.checkpoint_details, project_id, job_id
    )
    model_weight_format = checkpoint_details.checkpoints[0].model_weight_format

    base_model_id = checkpoint_details.base_model_id
    if deploy_config.model_name:
        model_name = deploy_config.model_name
    else:
        model_name = _get_model_name(model_weight_format, base_model_id)

    compute = _ensure_compute_spec(deploy_config.compute, remote_provider)

    runtime = _ensure_runtime_config(deploy_config.runtime)

    return DeployCheckpointsConfigComplete(
        checkpoint_details=checkpoint_details,
        model_name=model_name,
        runtime=runtime,
        compute=compute,
    )


def hydrate_checkpoint(
    job_id: str, checkpoint_id: str, checkpoint: dict, checkpoint_type: str
) -> Checkpoint:
    """
    Generic function to create a Checkpoint object for different model weight formats.
    This function can be extended to support additional checkpoint types beyond LoRA.
    """

    if checkpoint_type.lower() == ModelWeightsFormat.LORA.value:
        return hydrate_lora_checkpoint(job_id, checkpoint_id, checkpoint)
    elif checkpoint_type.lower() == ModelWeightsFormat.FULL.value:
        return hydrate_full_checkpoint(job_id, checkpoint_id, checkpoint)
    elif checkpoint_type.lower() == ModelWeightsFormat.WHISPER.value:
        return hydrate_whisper_checkpoint(job_id, checkpoint_id, checkpoint)
    else:
        raise ValueError(
            f"Unsupported checkpoint type: {checkpoint_type}. Contact Baseten for support with other checkpoint types."
        )


def _ensure_checkpoint_details(
    remote_provider: BasetenRemote,
    checkpoint_details: Optional[CheckpointList],
    project_id: Optional[str],
    job_id: Optional[str],
) -> CheckpointList:
    if checkpoint_details and checkpoint_details.checkpoints:
        # TODO: check here
        return _process_user_provided_checkpoints(checkpoint_details, remote_provider)
    else:
        return _prompt_user_for_checkpoint_details(
            remote_provider, checkpoint_details, project_id, job_id
        )


def _prompt_user_for_checkpoint_details(
    remote_provider: BasetenRemote,
    checkpoint_details: Optional[CheckpointList],
    project_id: Optional[str],
    job_id: Optional[str],
) -> CheckpointList:
    project_id, job_id = common.get_most_recent_job(remote_provider, project_id, job_id)
    response_checkpoints = _fetch_checkpoints(remote_provider, project_id, job_id)
    if not checkpoint_details:
        checkpoint_details = CheckpointList()

    # first, gather all checkpoint ids the user wants to deploy
    # allow the user to select a checkpoint
    checkpoint_ids = _get_checkpoint_ids_to_deploy(
        list(response_checkpoints.keys()), response_checkpoints
    )
    checkpoint_details.checkpoints = [
        _hydrate_checkpoints(job_id, checkpoint_id, response_checkpoints)
        for checkpoint_id in checkpoint_ids
    ]
    checkpoint_details.base_model_id = _get_base_model_id(
        checkpoint_details.base_model_id, response_checkpoints[checkpoint_ids[0]]
    )

    return checkpoint_details


def _process_user_provided_checkpoints(
    checkpoint_details: CheckpointList, remote_provider: BasetenRemote
) -> CheckpointList:
    # check if the user-provided checkpoint details are valid. Fill in missing values.
    checkpoints_by_training_job_id = {}
    for checkpoint in checkpoint_details.checkpoints:
        if checkpoint.training_job_id not in checkpoints_by_training_job_id:
            details = remote_provider.api.search_training_jobs(
                job_id=checkpoint.training_job_id
            )
            if len(details) == 0:
                raise click.UsageError(
                    f"Training job {checkpoint.training_job_id} not found."
                )
            job_response = details[0]
            project_id = job_response["training_project"]["id"]
            checkpoint_response = _fetch_checkpoints(
                remote_provider, project_id, checkpoint.training_job_id
            )
            # add to map of checkpoints by training job id
            checkpoints_by_training_job_id[checkpoint.training_job_id] = (
                checkpoint_response
            )
        checkpoint_response = checkpoints_by_training_job_id[checkpoint.training_job_id]
    return checkpoint_details


def _get_checkpoint_ids_to_deploy(
    checkpoint_id_options: List[str], response_checkpoints: OrderedDict[str, dict]
) -> List[str]:
    """Get checkpoint IDs to deploy based on user selection
    and validate the selected checkpoints.
    """
    if not checkpoint_id_options:
        raise click.UsageError("No checkpoints found for the training job.")

    # If only one checkpoint, return it directly
    if len(checkpoint_id_options) == 1:
        return [checkpoint_id_options[0]]

    checkpoint_ids = _select_multiple_checkpoints(checkpoint_id_options)
    _validate_selected_checkpoints(checkpoint_ids, response_checkpoints)
    return checkpoint_ids


def _select_multiple_checkpoints(checkpoint_id_options: List[str]) -> List[str]:
    """Select multiple checkpoints using interactive checkbox."""
    checkpoint_ids = inquirer.checkbox(
        message="Use spacebar to select/deselect checkpoints to deploy. Press enter when done.",
        choices=checkpoint_id_options,
    ).execute()

    if not checkpoint_ids:
        raise click.UsageError("At least one checkpoint must be selected.")

    return checkpoint_ids


def _ensure_compute_spec(
    compute: Optional[Compute], remote_provider: BasetenRemote
) -> Compute:
    if not compute:
        compute = Compute(cpu_count=0, memory="0Mi")
    compute = _get_accelerator_if_specified(compute, remote_provider)
    return compute


def _get_accelerator_if_specified(
    user_input: Optional[Compute], remote_provider: BasetenRemote
) -> Compute:
    if user_input and user_input.accelerator:
        return user_input

    # Fetch available instance types to get valid GPU options
    instance_types = remote_provider.api.get_instance_types()

    # Extract unique accelerator types from instance types
    accelerator_options = set()
    for it in instance_types:
        if it.gpu_type and it.gpu_count > 0:
            accelerator_options.add(it.gpu_type)

    # Convert to sorted list and add CPU option
    choices = sorted(list(accelerator_options)) + [None]

    if not choices or choices == [None]:
        console.print("No GPU instance types available, using CPU", style="yellow")
        return Compute(cpu_count=0, memory="0Mi", accelerator=None)

    # prompt user for accelerator
    gpu_type = inquirer.select(
        message="Select the GPU type to use for deployment. Select None for CPU.",
        choices=choices,
    ).execute()

    if gpu_type is None:
        return Compute(cpu_count=0, memory="0Mi", accelerator=None)

    # Get available counts for the selected GPU type
    available_counts = set()
    for it in instance_types:
        if it.gpu_type == gpu_type and it.gpu_count > 0:
            available_counts.add(it.gpu_count)
    if not available_counts:
        raise ValueError(
            f"No available counts for {gpu_type}. Reach out to Baseten for support if this persists."
        )

    if available_counts:
        count_choices = sorted(list(available_counts))
        count = inquirer.select(
            message=f"Select the number of {gpu_type} GPUs to use for deployment.",
            choices=count_choices,
            default=str(count_choices[0]),
        ).execute()
    else:
        count = inquirer.text(
            message=f"Enter the number of {gpu_type} accelerators to use for deployment.",
            default="1",
            validate=lambda x: x.isdigit() and int(x) > 0 and int(x) <= 8,
        ).execute()

    return Compute(
        cpu_count=0,
        memory="0Mi",
        accelerator=truss_config.AcceleratorSpec(
            accelerator=gpu_type.replace("-", "_"), count=int(count)
        ),
    )


def _get_base_model_id(user_input: Optional[str], checkpoint: dict) -> Optional[str]:
    if user_input:
        return user_input
        # prompt user for base model id
    base_model_id = None
    if base_model_id := checkpoint.get("base_model"):
        console.print(
            f"Inferring base model from checkpoint: {base_model_id}", style="yellow"
        )
    elif checkpoint.get("checkpoint_type") == ModelWeightsFormat.FULL.value.lower():
        return None
    elif checkpoint.get("checkpoint_type") == ModelWeightsFormat.WHISPER.value.lower():
        return None
    else:
        base_model_id = inquirer.text(message="Enter the base model id.").execute()
    if not base_model_id:
        raise click.UsageError(
            "Base model id is required. Please provide a base model id."
        )
    return base_model_id


def _ensure_runtime_config(
    runtime: Optional[DeployCheckpointsRuntime],
) -> DeployCheckpointsRuntime:
    if not runtime:
        runtime = DeployCheckpointsRuntime()
    if not runtime.environment_variables:
        # Prompt the user for the huggingface secret name as a default. There's much more we could
        # do here, but we're keeping it simple for now.
        hf_secret_name = get_hf_secret_name(
            runtime.environment_variables.get(HF_TOKEN_ENVVAR_NAME)
        )
        if hf_secret_name:
            runtime.environment_variables[HF_TOKEN_ENVVAR_NAME] = SecretReference(
                name=hf_secret_name
            )
    return runtime


def _fetch_checkpoints(
    remote_provider: BasetenRemote, project_id: str, job_id: str
) -> OrderedDict[str, dict]:
    console.print(f"Fetching checkpoints for training job {job_id}...")
    response = remote_provider.api.list_training_job_checkpoints(project_id, job_id)
    response_checkpoints = OrderedDict(
        (checkpoint["checkpoint_id"], checkpoint)
        for checkpoint in response["checkpoints"]
    )
    return response_checkpoints


def _hydrate_checkpoints(
    job_id: str, checkpoint_id: str, response_checkpoints: OrderedDict[str, dict]
) -> Checkpoint:
    if checkpoint_id == "latest":
        checkpoint_id = list(response_checkpoints.keys())[-1]
    checkpoint = response_checkpoints[checkpoint_id]
    checkpoint_type = str(checkpoint["checkpoint_type"])
    return hydrate_checkpoint(job_id, checkpoint_id, checkpoint, checkpoint_type)


def _validate_selected_checkpoints(
    checkpoint_ids: List[str], response_checkpoints: OrderedDict[str, dict]
) -> None:
    """
    Validates checkpoint model weight formats.
    - If the list contains any FULL checkpoint, only a single checkpoint is allowed.
    - If the list contains only LORA checkpoints, allow multiple checkpoints.
    """
    if (
        not checkpoint_ids
        or response_checkpoints[checkpoint_ids[0]].get("checkpoint_type") is None
    ):
        raise ValueError(
            "Unable to infer model weight format. Reach out to Baseten for support."
        )

    validation_rules = {
        ModelWeightsFormat.FULL.value: {
            "error_message": "Full checkpoints are not supported for multiple checkpoints. Please select a single checkpoint.",
            "reason": "vLLM does not support multiple checkpoints when any checkpoint is full model weights.",
        },
        ModelWeightsFormat.WHISPER.value: {
            "error_message": "Whisper checkpoints are not supported for multiple checkpoints. Please select a single checkpoint.",
            "reason": "vLLM does not support multiple checkpoints when any checkpoint is whisper model weights.",
        },
    }

    # Check each checkpoint type that has restrictions
    for checkpoint_type, rule in validation_rules.items():
        has_restricted_checkpoint = any(
            response_checkpoints[checkpoint_id].get("checkpoint_type")
            == checkpoint_type
            for checkpoint_id in checkpoint_ids
        )

        if has_restricted_checkpoint and len(checkpoint_ids) > 1:
            raise ValueError(rule["error_message"])


def get_hf_secret_name(user_input: Union[str, SecretReference, None]) -> str:
    """Get HuggingFace secret name from user input or prompt for it."""
    if not user_input:
        # prompt user for hf secret name
        hf_secret_name = inquirer.select(
            message="Enter the huggingface secret name.",
            choices=["hf_access_token", None, "custom"],
            default="hf_access_token",
        ).execute()
        if hf_secret_name == "custom":
            hf_secret_name = inquirer.text(
                message="Enter the huggingface secret name."
            ).execute()
        if not hf_secret_name:
            console.print("No hf secret name.", style="yellow")
        return hf_secret_name
    if isinstance(user_input, SecretReference):
        return user_input.name
    return user_input
