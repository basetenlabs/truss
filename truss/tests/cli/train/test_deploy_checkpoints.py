import os
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import truss_train.definitions as definitions
from truss.base import truss_config
from truss.cli.train.deploy_checkpoints import (
    _render_vllm_lora_truss_config,
    create_build_time_config,
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
        checkpoint_details=definitions.CheckpointList(
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
    mock_remote,
    deploy_checkpoints_mock_select,
    deploy_checkpoints_mock_text,
    deploy_checkpoints_mock_checkbox,
):
    # Create empty config
    empty_config = definitions.DeployCheckpointsConfig()

    # Call function under test
    result = prepare_checkpoint_deploy(
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
    mock_remote,
    deploy_checkpoints_mock_select,
    deploy_checkpoints_mock_text,
    deploy_checkpoints_mock_checkbox,
):
    # Create complete config with all fields specified
    complete_config = definitions.DeployCheckpointsConfig(
        checkpoint_details=definitions.CheckpointList(
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

    # open the config.yaml file and verify the tensor parallel size is 2
    # additional tests can be added to verify the config.yaml file is correct
    truss_cfg = truss_config.TrussConfig.from_yaml(
        Path(result.truss_directory, "config.yaml")
    )
    assert "--tensor-parallel-size 2" in truss_cfg.docker_server.start_command


def test_checkpoint_lora_rank_validation():
    """Test that Checkpoint accepts valid LoRA rank values."""
    valid_ranks = [8, 16, 32, 64, 128, 256, 320, 512]

    for rank in valid_ranks:
        checkpoint = definitions.Checkpoint(
            training_job_id="job123",
            id="checkpoint-1",
            name="test-checkpoint",
            lora_rank=rank,
        )
        assert checkpoint.lora_rank == rank

    invalid_ranks = [
        1,
        2,
        4,
        7,
        9,
        15,
        17,
        31,
        33,
        63,
        65,
        127,
        129,
        255,
        257,
        319,
        321,
        511,
        513,
        1000,
    ]
    for rank in invalid_ranks:
        with pytest.raises(ValueError, match=f"lora_rank \\({rank}\\) must be one of"):
            definitions.Checkpoint(
                training_job_id="job123",
                id="checkpoint-1",
                name="test-checkpoint",
                lora_rank=rank,
            )

    checkpoint = definitions.Checkpoint(
        training_job_id="job123",
        id="checkpoint-1",
        name="test-checkpoint",
        lora_rank=None,
    )
    assert checkpoint.lora_rank is None


def test_get_lora_rank():
    """Test that _get_lora_rank returns valid values from checkpoint response."""
    from truss.cli.train.deploy_checkpoints import _get_lora_rank

    # Test with valid rank from API
    checkpoint_resp = {"lora_adapter_config": {"r": 64}}
    assert _get_lora_rank(checkpoint_resp) == 64
    # Test with missing lora_adapter_config (should use DEFAULT_LORA_RANK)
    checkpoint_resp = {}
    assert _get_lora_rank(checkpoint_resp) == 16  # DEFAULT_LORA_RANK
    # Test with missing 'r' field (should use DEFAULT_LORA_RANK)
    checkpoint_resp = {"lora_adapter_config": {}}
    assert _get_lora_rank(checkpoint_resp) == 16  # DEFAULT_LORA_RANK
    # Test with invalid rank from API
    checkpoint_resp = {"lora_adapter_config": {"r": 1}}
    with pytest.raises(
        ValueError,
        match=re.escape("LoRA rank 1 from checkpoint is not in allowed values"),
    ):
        _get_lora_rank(checkpoint_resp)
    # Test with another invalid rank
    checkpoint_resp = {"lora_adapter_config": {"r": 1000}}
    with pytest.raises(
        ValueError,
        match=re.escape("LoRA rank 1000 from checkpoint is not in allowed values"),
    ):
        _get_lora_rank(checkpoint_resp)


def test_create_build_time_config(tmp_path):
    """Test that create_build_time_config properly creates a build-time config file."""
    # Create a temporary directory structure
    context_path = tmp_path / "context"
    context_path.mkdir()

    # Create a sample config.yaml file with runtime-only attributes
    config_data = {
        "model_name": "test-model",
        "docker_server": {
            "start_command": "python server.py --port 8000",
            "server_port": 8000,
            "predict_endpoint": "/predict",
            "readiness_endpoint": "/health",
            "liveness_endpoint": "/health",
        },
        "training_checkpoints": {
            "download_folder": "/tmp/checkpoints",
            "checkpoints": [{"id": "checkpoint-1", "name": "checkpoint-1"}],
        },
        "resources": {
            "cpu": "1000m",
            "memory": "2Gi",
            "accelerator": {"accelerator": "H100", "count": 1},
        },
        "environment_variables": {"HF_TOKEN": "secret_token"},
    }

    # Write the config file
    config_path = context_path / "config.yaml"
    import yaml

    with open(config_path, "w") as f:
        yaml.dump(config_data, f)

    # Call the function under test
    create_build_time_config(context_path)

    # Verify the build-time config file was created
    build_time_config_path = context_path / "config_build_time.yaml"
    assert build_time_config_path.exists()

    # Load and verify the build-time config
    build_time_config = truss_config.TrussConfig.from_yaml(build_time_config_path)

    # Verify that some runtime-only attributes are preserved (not yet supported for buildless)
    assert build_time_config.model_name == "test-model"  # Should be preserved
    assert build_time_config.resources.cpu == "1000m"  # Should be preserved
    assert build_time_config.resources.memory == "2Gi"  # Should be preserved
    assert (
        build_time_config.environment_variables["HF_TOKEN"] == "secret_token"
    )  # Should be preserved

    # Verify that runtime-only attributes are excluded
    assert build_time_config.training_checkpoints is None  # Should be set to None
    assert (
        build_time_config.docker_server.start_command == ""
    )  # Should be set to empty string
