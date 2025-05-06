import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

import truss_train.definitions as definitions
from truss.base import truss_config
from truss.cli.train.deploy_checkpoints import (
    _render_vllm_lora_truss_config,
    prepare_checkpoint_deploy,
)
from truss.cli.train.types import (
    DeployCheckpointsConfigComplete,
    PrepareCheckpointResult,
)


@pytest.fixture
def mock_remote():
    mock = MagicMock()
    mock.api.search_training_jobs.return_value = [
        {"training_project": {"id": "project123"}}
    ]
    mock.api.list_training_job_checkpoints.return_value = {
        "checkpoints": [
            {
                "checkpoint_id": "checkpoint-1",
                "base_model": "google/gemma-3-27b-it",
                "lora_adapter_config": {"r": 16},
            }
        ]
    }
    return mock


@pytest.fixture
def mock_console():
    return Console()


@pytest.fixture
def create_mock_prompt():
    def _create_mock_prompt(return_value):
        mock_prompt = MagicMock()
        mock_prompt.execute.return_value = return_value
        return mock_prompt

    return _create_mock_prompt


@pytest.fixture
def deploy_checkpoints_mock_select(create_mock_prompt):
    with patch("truss.cli.train.deploy_checkpoints.inquirer.select") as mock:
        mock.side_effect = lambda message, **kwargs: create_mock_prompt(
            "H100"
            if "GPU type" in message
            else "hf_access_token"
            if "huggingface secret name" in message
            else None
        )
        yield mock


@pytest.fixture
def deploy_checkpoints_mock_text(create_mock_prompt):
    with patch("truss.cli.train.deploy_checkpoints.inquirer.text") as mock:
        mock.side_effect = lambda message, **kwargs: create_mock_prompt(
            "4" if "number of accelerators" in message else None
        )
        yield mock


@pytest.fixture
def deploy_checkpoints_mock_checkbox(create_mock_prompt):
    with patch("truss.cli.train.deploy_checkpoints.inquirer.checkbox") as mock:
        mock.side_effect = lambda message, **kwargs: create_mock_prompt(
            ["checkpoint-1"] if "Select the checkpoint" in message else None
        )
        yield mock


def test_render_vllm_lora_truss_config():
    deploy_config = DeployCheckpointsConfigComplete(
        checkpoint_details=definitions.CheckpointDetails(
            checkpoints=[
                definitions.Checkpoint(
                    id="checkpoint-1",
                    name="checkpoint-1",
                    lora_rank=16,
                    training_job_id="kowpeqj",
                )
            ],
            base_model_id="google/gemma-3-27b-it",
        ),
        model_name="gemma-3-27b-it-vLLM-LORA",
        compute=definitions.Compute(
            accelerator=truss_config.AcceleratorSpec(accelerator="H100", count=4)
        ),
        runtime=definitions.DeployCheckpointsRuntime(
            environment_variables={
                "HF_TOKEN": definitions.SecretReference(name="hf_access_token")
            }
        ),
        deployment_name="gemma-3-27b-it-vLLM-LORA",
    )
    rendered_truss = _render_vllm_lora_truss_config(deploy_config)
    test_truss = truss_config.TrussConfig.from_yaml(
        Path(
            os.path.dirname(__file__),
            "resources/test_deploy_from_checkpoint_config.yml",
        )
    )
    assert test_truss.model_name == rendered_truss.model_name
    assert (
        test_truss.training_checkpoints.checkpoints[0].id
        == rendered_truss.training_checkpoints.checkpoints[0].id
    )
    assert (
        test_truss.training_checkpoints.checkpoints[0].name
        == rendered_truss.training_checkpoints.checkpoints[0].name
    )
    assert (
        test_truss.docker_server.start_command
        == rendered_truss.docker_server.start_command
    )
    assert test_truss.resources.accelerator == rendered_truss.resources.accelerator
    assert test_truss.secrets == rendered_truss.secrets
    assert test_truss.training_checkpoints == rendered_truss.training_checkpoints


def test_prepare_checkpoint_deploy_empty_config(
    mock_console,
    mock_remote,
    deploy_checkpoints_mock_select,
    deploy_checkpoints_mock_text,
    deploy_checkpoints_mock_checkbox,
):
    # Create empty config
    empty_config = definitions.DeployCheckpointsConfig()

    # Call function under test
    result = prepare_checkpoint_deploy(
        console=mock_console,
        remote_provider=mock_remote,
        checkpoint_deploy_config=empty_config,
        project_id="project123",
        job_id="job123",
    )

    assert isinstance(result, PrepareCheckpointResult)
    assert result.checkpoint_deploy_config.model_name == "gemma-3-27b-it-vLLM-LORA"
    assert (
        result.checkpoint_deploy_config.checkpoint_details.base_model_id
        == "google/gemma-3-27b-it"
    )
    assert len(result.checkpoint_deploy_config.checkpoint_details.checkpoints) == 1
    assert (
        result.checkpoint_deploy_config.checkpoint_details.checkpoints[0].id
        == "checkpoint-1"
    )
    assert (
        result.checkpoint_deploy_config.checkpoint_details.checkpoints[0].lora_rank
        == 16
    )
    assert result.checkpoint_deploy_config.compute.accelerator.accelerator == "H100"
    assert result.checkpoint_deploy_config.compute.accelerator.count == 4
    assert (
        result.checkpoint_deploy_config.runtime.environment_variables["HF_TOKEN"].name
        == "hf_access_token"
    )


def test_prepare_checkpoint_deploy_complete_config(
    mock_console,
    mock_remote,
    deploy_checkpoints_mock_select,
    deploy_checkpoints_mock_text,
    deploy_checkpoints_mock_checkbox,
):
    # Create complete config with all fields specified
    complete_config = definitions.DeployCheckpointsConfig(
        checkpoint_details=definitions.CheckpointDetails(
            checkpoints=[
                definitions.Checkpoint(
                    id="checkpoint-1",
                    name="my-checkpoint",
                    lora_rank=32,
                    training_job_id="job123",
                )
            ],
            base_model_id="google/gemma-3-27b-it",
        ),
        model_name="my-custom-model",
        deployment_name="my-deployment",
        compute=definitions.Compute(
            accelerator=truss_config.AcceleratorSpec(accelerator="A100", count=2)
        ),
        runtime=definitions.DeployCheckpointsRuntime(
            environment_variables={
                "HF_TOKEN": definitions.SecretReference(name="my_custom_secret"),
                "CUSTOM_VAR": "custom_value",
            }
        ),
    )

    # Call function under test
    result = prepare_checkpoint_deploy(
        console=mock_console,
        remote_provider=mock_remote,
        checkpoint_deploy_config=complete_config,
        project_id="project123",
        job_id="job123",
    )

    # Verify result
    assert isinstance(result, PrepareCheckpointResult)

    # Verify no prompts were called
    deploy_checkpoints_mock_select.assert_not_called()
    deploy_checkpoints_mock_text.assert_not_called()
    deploy_checkpoints_mock_checkbox.assert_not_called()

    # Verify config values were preserved
    config = result.checkpoint_deploy_config
    assert config.model_name == "my-custom-model"
    assert config.deployment_name == "my-deployment"

    # Verify checkpoint details
    assert config.checkpoint_details.base_model_id == "google/gemma-3-27b-it"
    assert len(config.checkpoint_details.checkpoints) == 1
    checkpoint = config.checkpoint_details.checkpoints[0]
    assert checkpoint.id == "checkpoint-1"
    assert checkpoint.name == "my-checkpoint"
    assert checkpoint.lora_rank == 32
    assert checkpoint.training_job_id == "job123"

    # Verify compute config
    assert config.compute.accelerator.accelerator == "A100"
    assert config.compute.accelerator.count == 2

    # Verify runtime config
    env_vars = config.runtime.environment_variables
    assert env_vars["HF_TOKEN"].name == "my_custom_secret"
    assert env_vars["CUSTOM_VAR"] == "custom_value"
