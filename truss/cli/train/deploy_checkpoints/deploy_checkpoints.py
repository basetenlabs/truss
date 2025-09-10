import re
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import List, Optional, Union

import rich_click as click
from InquirerPy import inquirer

from truss.base import truss_config
from truss.cli.train import common
from truss.cli.train.types import (
    DeployCheckpointsConfigComplete,
    PrepareCheckpointResult,
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

from .deploy_full_checkpoints import (
    hydrate_full_checkpoint,
    render_vllm_full_truss_config,
)
from .deploy_lora_checkpoints import (
    hydrate_lora_checkpoint,
    render_vllm_lora_truss_config,
)
from .deploy_whisper_checkpoints import (
    hydrate_whisper_checkpoint,
    render_vllm_whisper_truss_config,
)

HF_TOKEN_ENVVAR_NAME = "HF_TOKEN"
# If we change this, make sure to update the logic in backend codebase
CHECKPOINT_PATTERN = re.compile(r".*checkpoint-\d+(?:-\d+)?$")
ALLOWED_DEPLOYMENT_NAMES = re.compile(r"^[0-9a-zA-Z_\-\.]*$")


def prepare_checkpoint_deploy(
    remote_provider: BasetenRemote,
    checkpoint_deploy_config: DeployCheckpointsConfig,
    project_id: Optional[str],
    job_id: Optional[str],
) -> PrepareCheckpointResult:
    checkpoint_deploy_config = _hydrate_deploy_config(
        checkpoint_deploy_config, remote_provider, project_id, job_id
    )
    rendered_truss = _render_truss_config_for_checkpoint_deployment(
        checkpoint_deploy_config
    )
    truss_directory = Path(tempfile.mkdtemp())
    truss_config_path = truss_directory / "config.yaml"
    rendered_truss.write_to_yaml_file(truss_config_path)
    console.print(rendered_truss, style="green")
    console.print(f"Writing truss config to {truss_config_path}", style="yellow")
    return PrepareCheckpointResult(
        truss_directory=truss_directory,
        checkpoint_deploy_config=checkpoint_deploy_config,
    )


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

    model_name = inquirer.text(
        message=f"Enter the model name for your {model_weight_format.value} model.",
        validate=lambda s: s and s.strip(),
        default=default,
    ).execute()

    if model_weight_format == ModelWeightsFormat.FULL:
        model_name += "-vLLM-Full"
    elif model_weight_format == ModelWeightsFormat.LORA:
        model_name += "-vLLM-LORA"
    return model_name


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

    compute = _ensure_compute_spec(deploy_config.compute)

    runtime = _ensure_runtime_config(deploy_config.runtime)
    deployment_name = _ensure_deployment_name(
        deploy_config.deployment_name, checkpoint_details.checkpoints
    )

    return DeployCheckpointsConfigComplete(
        checkpoint_details=checkpoint_details,
        model_name=model_name,
        deployment_name=deployment_name,
        runtime=runtime,
        compute=compute,
        model_weight_format=model_weight_format.to_truss_config(),  # type: ignore[attr-defined]
    )


def _ensure_deployment_name(
    deploy_config_deployment_name: Optional[str], checkpoints: List[Checkpoint]
) -> str:
    if deploy_config_deployment_name:
        return deploy_config_deployment_name

    default_deployment_name = "checkpoint"

    if checkpoints and checkpoints[0].paths:
        first_checkpoint_name = checkpoints[0].paths[0].strip("/").split("/")[-1]

        if ALLOWED_DEPLOYMENT_NAMES.match(first_checkpoint_name):
            # Allow autoincrementing if the checkpoint matches both regexes
            if (
                CHECKPOINT_PATTERN.match(first_checkpoint_name)
                and len(checkpoints) == 1
            ):
                return first_checkpoint_name

    # If no valid autoincrementing checkpoint name is found, prompt the user
    deployment_name = inquirer.text(
        message="Enter the deployment name.", default=default_deployment_name
    ).execute()

    if not deployment_name:
        raise click.UsageError("Deployment name is required.")

    return deployment_name


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


def _render_truss_config_for_checkpoint_deployment(
    checkpoint_deploy: DeployCheckpointsConfigComplete,
) -> truss_config.TrussConfig:
    """
    Render truss config for checkpoint deployment.
    Currently supports LoRA checkpoints via vLLM, but can be extended for other formats.
    """
    # Delegate to specific rendering function based on model weight format
    if checkpoint_deploy.model_weight_format == ModelWeightsFormat.LORA:
        return render_vllm_lora_truss_config(checkpoint_deploy)
    elif checkpoint_deploy.model_weight_format == ModelWeightsFormat.FULL:
        return render_vllm_full_truss_config(checkpoint_deploy)
    elif checkpoint_deploy.model_weight_format == ModelWeightsFormat.WHISPER:
        return render_vllm_whisper_truss_config(checkpoint_deploy)
    else:
        raise ValueError(
            f"Unsupported model weight format: {checkpoint_deploy.model_weight_format}. Please upgrade to the latest Truss version to access the latest supported formats. Contact Baseten if you would like us to support additional formats."
        )


def _ensure_checkpoint_details(
    remote_provider: BasetenRemote,
    checkpoint_details: Optional[CheckpointList],
    project_id: Optional[str],
    job_id: Optional[str],
) -> CheckpointList:
    if checkpoint_details and checkpoint_details.checkpoints:
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


def _ensure_compute_spec(compute: Optional[Compute]) -> Compute:
    if not compute:
        compute = Compute(cpu_count=0, memory="0Mi")
    compute.accelerator = _get_accelerator_if_specified(compute.accelerator)
    return compute


def _get_accelerator_if_specified(
    user_input: Optional[truss_config.AcceleratorSpec],
) -> Optional[truss_config.AcceleratorSpec]:
    if user_input:
        return user_input
    # prompt user for accelerator
    gpu_type = inquirer.select(
        message="Select the GPU type to use for deployment. Select None for CPU.",
        choices=[x.value for x in truss_config.Accelerator] + [None],
    ).execute()
    if gpu_type is None:
        return None
    count = inquirer.text(
        message="Enter the number of accelerators to use for deployment.",
        default="1",
        validate=lambda x: x.isdigit() and int(x) > 0 and int(x) <= 8,
    ).execute()
    return truss_config.AcceleratorSpec(accelerator=gpu_type, count=int(count))


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
