import os
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import List, Optional, Union

import rich
import rich_click as click
from InquirerPy import inquirer
from jinja2 import Template
from rich.console import Console
from rich.text import Text

from truss.base import truss_config
from truss.cli.common import get_most_recent_job
from truss.cli.train.types import (
    DeployCheckpointsConfigComplete,
    PrepareCheckpointResult,
)
from truss.remote.baseten.remote import BasetenRemote
from truss_train.definitions import (
    DEFAULT_LORA_RANK,
    Checkpoint,
    CheckpointDetails,
    Compute,
    DeployCheckpointsConfig,
    DeployCheckpointsRuntime,
    SecretReference,
)

VLLM_LORA_START_COMMAND = Template(
    'sh -c "{%if envvars %}{{ envvars }} {% endif %}vllm serve {{ base_model_id }}'
    + " --port 8000"
    + " --tensor-parallel-size 4"
    + " --enable-lora"
    + " --max-lora-rank {{ max_lora_rank }}"
    + " --dtype bfloat16"
    + ' --lora-modules {{ lora_modules }}"'
)

HF_TOKEN_ENVVAR_NAME = "HF_TOKEN"


def prepare_checkpoint_deploy(
    console: Console,
    remote_provider: BasetenRemote,
    checkpoint_deploy_config: DeployCheckpointsConfig,
    project_id: Optional[str],
    job_id: Optional[str],
) -> PrepareCheckpointResult:
    checkpoint_deploy_config = _hydrate_deploy_config(
        console, checkpoint_deploy_config, remote_provider, project_id, job_id
    )
    rendered_truss = _render_vllm_lora_truss_config(checkpoint_deploy_config)
    truss_directory = Path(
        tempfile.mkdtemp(suffix=f"{checkpoint_deploy_config.deployment_name}")
    )
    truss_config_path = truss_directory / "config.yaml"
    rendered_truss.write_to_yaml_file(truss_config_path)
    console.print(rendered_truss, style="green")
    console.print(f"Writing truss config to {truss_config_path}", style="yellow")
    return PrepareCheckpointResult(
        truss_directory=truss_directory,
        checkpoint_deploy_config=checkpoint_deploy_config,
    )


def _hydrate_deploy_config(
    console: Console,
    deploy_config: DeployCheckpointsConfig,
    remote_provider: BasetenRemote,
    project_id: Optional[str],
    job_id: Optional[str],
) -> DeployCheckpointsConfigComplete:
    checkpoint_details = _get_checkpoint_details(
        console, remote_provider, deploy_config.checkpoint_details, project_id, job_id
    )
    base_model_id = checkpoint_details.base_model_id
    if not base_model_id:
        raise ValueError(
            "Unable to infer base model id. Reach out to Baseten for support."
        )
    compute = _get_compute(deploy_config.compute)
    model_name = (
        deploy_config.model_name or f"{base_model_id.split('/')[-1]}-vLLM-LORA"  #
    )
    runtime = _get_runtime(console, deploy_config.runtime)
    deployment_name = (
        deploy_config.deployment_name or checkpoint_details.checkpoints[0].id
    )
    return DeployCheckpointsConfigComplete(
        checkpoint_details=checkpoint_details,
        model_name=model_name,
        deployment_name=deployment_name,
        runtime=runtime,
        compute=compute,
    )


def _render_vllm_lora_truss_config(
    checkpoint_deploy: DeployCheckpointsConfigComplete,
) -> truss_config.TrussConfig:
    truss_deploy_config = truss_config.TrussConfig.from_yaml(
        Path(os.path.dirname(__file__), "deploy_from_checkpoint_config.yml")
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
        # this is a quirk of serving vllm with secrets - we need to export the secret by cat-ing it
        if isinstance(value, SecretReference):
            truss_deploy_config.secrets[value.name] = "set token in baseten workspace"
            start_command_envvars = f"{key}=$(cat /secrets/{value.name})"

    checkpoint_parts = []
    for truss_checkpoint in truss_deploy_config.training_checkpoints.checkpoints:  # type: ignore
        ckpt_path = Path(
            truss_deploy_config.training_checkpoints.download_folder,  # type: ignore
            truss_checkpoint.id,
        )
        checkpoint_parts.append(f"{truss_checkpoint.name}={ckpt_path}")
    checkpoint_str = " ".join(checkpoint_parts)
    max_lora_rank = max(
        [
            checkpoint.lora_rank or DEFAULT_LORA_RANK
            for checkpoint in checkpoint_deploy.checkpoint_details.checkpoints
        ]
    )

    start_command_args = {
        "base_model_id": checkpoint_deploy.checkpoint_details.base_model_id,
        "lora_modules": checkpoint_str,
        "envvars": start_command_envvars,
        "max_lora_rank": max_lora_rank,
    }
    truss_deploy_config.docker_server.start_command = VLLM_LORA_START_COMMAND.render(
        **start_command_args
    )
    return truss_deploy_config


def _get_checkpoint_details(
    console: Console,
    remote_provider: BasetenRemote,
    checkpoint_details: Optional[CheckpointDetails],
    project_id: Optional[str],
    job_id: Optional[str],
) -> CheckpointDetails:
    if checkpoint_details and checkpoint_details.checkpoints:
        return _process_user_provided_checkpoints(checkpoint_details, remote_provider)
    else:
        return _prompt_user_for_checkpoint_details(
            console, remote_provider, checkpoint_details, project_id, job_id
        )


def _prompt_user_for_checkpoint_details(
    console: Console,
    remote_provider: BasetenRemote,
    checkpoint_details: Optional[CheckpointDetails],
    project_id: Optional[str],
    job_id: Optional[str],
) -> CheckpointDetails:
    project_id, job_id = get_most_recent_job(
        console, remote_provider, project_id, job_id
    )
    response_checkpoints = _fetch_checkpoints(remote_provider, project_id, job_id)
    if not checkpoint_details:
        checkpoint_details = CheckpointDetails()

    # first, gather all checkpoint ids the user wants to deploy
    # allow the user to select a checkpoint
    checkpoint_ids = _get_checkpoint_ids_to_deploy(list(response_checkpoints.keys()))
    checkpoint_details.checkpoints = [
        _hydrate_checkpoints(job_id, checkpoint_id, response_checkpoints)
        for checkpoint_id in checkpoint_ids
    ]
    checkpoint_details.base_model_id = _get_base_model_id(
        checkpoint_details.base_model_id, response_checkpoints[checkpoint_ids[0]]
    )
    return checkpoint_details


def _process_user_provided_checkpoints(
    checkpoint_details: CheckpointDetails, remote_provider: BasetenRemote
) -> CheckpointDetails:
    # check if the user-provided checkpoint details are valid. Fill in missing values.
    checkpoints_by_training_job_id = {}
    for checkpoint in checkpoint_details.checkpoints:
        if checkpoint.training_job_id not in checkpoints_by_training_job_id:
            details = remote_provider.api.search_training_jobs(
                job_id=checkpoint.training_job_id
            )
            if len(details) == 0:
                raise click.UsageError(
                    f"Training job {checkpoint.training_job_id} specified by checkpoint {checkpoint.id} not found."
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
        if checkpoint.id not in checkpoint_response:
            raise click.UsageError(f"Checkpoint {checkpoint.id} not found.")
        if not checkpoint.name:
            checkpoint.name = checkpoint.id
        if not checkpoint.lora_rank:
            checkpoint.lora_rank = _get_lora_rank(checkpoint_response[checkpoint.id])
    return checkpoint_details


def _get_checkpoint_ids_to_deploy(checkpoint_id_options: List[str]) -> str:
    if len(checkpoint_id_options) == 0:
        raise click.UsageError("No checkpoints found for the training job.")
    if len(checkpoint_id_options) > 1:
        checkpoint_ids = inquirer.checkbox(
            message="Select the checkpoint to deploy. Use spacebar to select/deselect.",
            choices=checkpoint_id_options,
        ).execute()
        if not checkpoint_ids:
            raise click.UsageError("At least one checkpoint must be selected.")
    else:
        checkpoint_ids = [checkpoint_id_options[0]]
    return checkpoint_ids


def _hydrate_checkpoints(
    job_id: str, checkpoint_id: str, response_checkpoints: OrderedDict[str, dict]
) -> Checkpoint:
    if checkpoint_id == "latest":
        checkpoint_id = list(response_checkpoints.keys())[-1]
    checkpoint = response_checkpoints[checkpoint_id]
    return Checkpoint(
        training_job_id=job_id,
        id=checkpoint_id,
        name=checkpoint_id,
        lora_rank=_get_lora_rank(checkpoint),
    )


def _get_lora_rank(checkpoint_resp: dict) -> int:
    lora_adapter_config = checkpoint_resp.get("lora_adapter_config") or {}
    return lora_adapter_config.get("r") or DEFAULT_LORA_RANK


def _get_hf_secret_name(
    console: Console, user_input: Union[str, SecretReference, None]
) -> str:
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


def _get_compute(compute: Optional[Compute]) -> Compute:
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


def _get_base_model_id(user_input: Optional[str], checkpoint: dict) -> str:
    if user_input:
        return user_input
        # prompt user for base model id
    base_model_id = None
    if base_model_id := checkpoint.get("base_model"):
        rich.print(
            Text(
                f"Inferring base model from checkpoint: {base_model_id}", style="yellow"
            )
        )
    else:
        base_model_id = inquirer.text(message="Enter the base model id.").execute()
    if not base_model_id:
        raise click.UsageError(
            "Base model id is required. Please provide a base model id."
        )
    return base_model_id


def _get_runtime(
    console: Console, runtime: Optional[DeployCheckpointsRuntime]
) -> DeployCheckpointsRuntime:
    if not runtime:
        runtime = DeployCheckpointsRuntime()
    if not runtime.environment_variables:
        # Prompt the user for the huggingface secret name as a default. There's much more we could
        # do here, but we're keeping it simple for now.
        hf_secret_name = _get_hf_secret_name(
            console, runtime.environment_variables.get(HF_TOKEN_ENVVAR_NAME)
        )
        runtime.environment_variables[HF_TOKEN_ENVVAR_NAME] = SecretReference(
            name=hf_secret_name
        )
    return runtime


def _fetch_checkpoints(
    remote_provider: BasetenRemote, project_id: str, job_id: str
) -> OrderedDict[str, dict]:
    rich.print(f"Fetching checkpoints for training job {job_id}...")
    response = remote_provider.api.list_training_job_checkpoints(project_id, job_id)
    response_checkpoints = OrderedDict(
        (checkpoint["checkpoint_id"], checkpoint)
        for checkpoint in response["checkpoints"]
    )
    return response_checkpoints
