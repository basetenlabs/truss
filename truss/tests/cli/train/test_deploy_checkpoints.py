import os
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import truss_train.definitions as definitions
from truss.base import truss_config
from truss.cli.train.deploy_checkpoints import prepare_checkpoint_deploy
from truss.cli.train.deploy_checkpoints.deploy_checkpoints import (
    _render_truss_config_for_checkpoint_deployment,
    hydrate_checkpoint,
)
from truss.cli.train.deploy_checkpoints.deploy_lora_checkpoints import (
    START_COMMAND_ENVVAR_NAME,
    _get_lora_rank,
    hydrate_lora_checkpoint,
    render_vllm_lora_truss_config,
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
                "checkpoint_type": "lora",
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
    # Only patch the module that actually imports inquirer
    with patch(
        "truss.cli.train.deploy_checkpoints.deploy_checkpoints.inquirer.select"
    ) as generic_mock:

        def mock_select_side_effect(message, **kwargs):
            return create_mock_prompt(
                "LoRA"
                if "model weight format" in message
                else "H100"
                if "GPU type" in message
                else "hf_access_token"
                if "huggingface secret name" in message
                else None
            )

        generic_mock.side_effect = mock_select_side_effect
        yield generic_mock


@pytest.fixture
def deploy_checkpoints_mock_text(create_mock_prompt):
    with patch(
        "truss.cli.train.deploy_checkpoints.deploy_checkpoints.inquirer.text"
    ) as mock:
        mock.side_effect = lambda message, **kwargs: create_mock_prompt(
            "4"
            if "number of accelerators" in message
            else "test-deployment"
            if "deployment name" in message
            else None
        )
        yield mock


@pytest.fixture
def deploy_checkpoints_mock_checkbox(create_mock_prompt):
    with patch(
        "truss.cli.train.deploy_checkpoints.deploy_checkpoints.inquirer.checkbox"
    ) as mock:
        mock.side_effect = lambda message, **kwargs: create_mock_prompt(
            ["checkpoint-1"] if "Select the checkpoint" in message else None
        )
        yield mock


def test_render_truss_config_for_checkpoint_deployment():
    deploy_config = DeployCheckpointsConfigComplete(
        checkpoint_details=definitions.CheckpointList(
            checkpoints=[
                definitions.LoRACheckpoint(
                    training_job_id="job123",
                    paths=["rank-0/checkpoint-1/"],
                    model_weight_format=definitions.ModelWeightsFormat.LORA,
                    lora_details=definitions.LoRADetails(rank=16),
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
        model_weight_format=truss_config.ModelWeightsFormat.LORA,
    )
    rendered_truss = _render_truss_config_for_checkpoint_deployment(deploy_config)
    test_truss = truss_config.TrussConfig.from_yaml(
        Path(
            os.path.dirname(__file__),
            "resources/test_deploy_from_checkpoint_config.yml",
        )
    )
    assert test_truss.model_name == rendered_truss.model_name
    assert (
        test_truss.training_checkpoints.artifact_references[0].paths[0]
        == rendered_truss.training_checkpoints.artifact_references[0].paths[0]
    )
    assert (
        test_truss.training_checkpoints.artifact_references[0].paths
        == rendered_truss.training_checkpoints.artifact_references[0].paths
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
    checkpoint = result.checkpoint_deploy_config.checkpoint_details.checkpoints[0]
    assert checkpoint.training_job_id == "job123"
    assert isinstance(checkpoint, definitions.LoRACheckpoint)
    assert checkpoint.lora_details.rank == 16
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
                definitions.LoRACheckpoint(
                    training_job_id="job123",
                    paths=["job123/rank-0/checkpoint-1/"],
                    model_weight_format=definitions.ModelWeightsFormat.LORA,
                    lora_details=definitions.LoRADetails(rank=32),
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
    assert checkpoint.training_job_id == "job123"
    assert checkpoint.model_weight_format == definitions.ModelWeightsFormat.LORA
    assert isinstance(checkpoint, definitions.LoRACheckpoint)
    assert checkpoint.lora_details.rank == 32

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
    # Check that the start command is now the environment variable reference
    assert truss_cfg.docker_server.start_command == "%(ENV_BT_DOCKER_SERVER_START_CMD)s"
    # Check that the actual start command with tensor parallel size is in the environment variable
    assert (
        "--tensor-parallel-size 2"
        in truss_cfg.environment_variables["BT_DOCKER_SERVER_START_CMD"]
    )


def test_checkpoint_lora_rank_validation():
    """Test that LoRACheckpoint accepts valid LoRA rank values."""
    valid_ranks = [8, 16, 32, 64, 128, 256, 320, 512]

    for rank in valid_ranks:
        checkpoint = definitions.LoRACheckpoint(
            training_job_id="job123",
            paths=["job123/rank-0/checkpoint-1/"],
            model_weight_format=definitions.ModelWeightsFormat.LORA,
            lora_details=definitions.LoRADetails(rank=rank),
        )
        assert checkpoint.lora_details.rank == rank

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
            definitions.LoRACheckpoint(
                training_job_id="job123",
                paths=["job123/rank-0/checkpoint-1/"],
                model_weight_format=definitions.ModelWeightsFormat.LORA,
                lora_details=definitions.LoRADetails(rank=rank),
            )


def test_get_lora_rank():
    """Test that get_lora_rank returns valid values from checkpoint response."""
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


def test_hydrate_lora_checkpoint():
    """Test that hydrate_lora_checkpoint creates proper LoRACheckpoint objects."""
    job_id = "test_job_123"
    checkpoint_id = "checkpoint-456"
    checkpoint_data = {
        "lora_adapter_config": {"r": 64},
        "base_model": "google/gemma-3-27b-it",
        "checkpoint_type": "lora",
    }

    result = hydrate_lora_checkpoint(job_id, checkpoint_id, checkpoint_data)

    assert isinstance(result, definitions.LoRACheckpoint)
    assert result.training_job_id == job_id
    assert result.lora_details.rank == 64
    assert len(result.paths) == 1
    assert result.paths[0] == f"rank-0/{checkpoint_id}/"


def test_hydrate_checkpoint_dispatcher():
    """Test that hydrate_checkpoint properly dispatches to the right function based on checkpoint type."""
    job_id = "test_job_123"
    checkpoint_id = "checkpoint-456"
    checkpoint_data = {
        "lora_adapter_config": {"r": 32},
        "base_model": "google/gemma-3-27b-it",
        "checkpoint_type": "lora",
    }

    # Test LoRA checkpoint type
    result = hydrate_checkpoint(job_id, checkpoint_id, checkpoint_data, "lora")
    assert isinstance(result, definitions.LoRACheckpoint)
    assert result.lora_details.rank == 32

    # Test uppercase LoRA checkpoint type
    result = hydrate_checkpoint(job_id, checkpoint_id, checkpoint_data, "LORA")
    assert isinstance(result, definitions.LoRACheckpoint)
    assert result.lora_details.rank == 32

    # Test unsupported checkpoint type
    with pytest.raises(ValueError, match="Unsupported checkpoint type: unsupported"):
        hydrate_checkpoint(job_id, checkpoint_id, checkpoint_data, "unsupported")


def test_render_vllm_lora_truss_config():
    """Test that render_vllm_lora_truss_config creates proper TrussConfig for LoRA deployments."""
    deploy_config = DeployCheckpointsConfigComplete(
        checkpoint_details=definitions.CheckpointList(
            checkpoints=[
                definitions.LoRACheckpoint(
                    training_job_id="job123",
                    paths=["rank-0/checkpoint-1/"],
                    model_weight_format=definitions.ModelWeightsFormat.LORA,
                    lora_details=definitions.LoRADetails(rank=64),
                )
            ],
            base_model_id="google/gemma-3-27b-it",
        ),
        model_name="test-lora-model",
        compute=definitions.Compute(
            accelerator=truss_config.AcceleratorSpec(accelerator="H100", count=2)
        ),
        runtime=definitions.DeployCheckpointsRuntime(
            environment_variables={
                "HF_TOKEN": definitions.SecretReference(name="hf_token")
            }
        ),
        deployment_name="test-deployment",
        model_weight_format=truss_config.ModelWeightsFormat.LORA,
    )

    result = render_vllm_lora_truss_config(deploy_config)

    expected_vllm_command = 'sh -c "HF_TOKEN=$(cat /secrets/hf_token) vllm serve google/gemma-3-27b-it --port 8000 --tensor-parallel-size 2 --enable-lora --max-lora-rank 64 --dtype bfloat16 --lora-modules job123=/tmp/training_checkpoints/job123/rank-0/checkpoint-1"'

    assert isinstance(result, truss_config.TrussConfig)
    assert result.model_name == "test-lora-model"
    assert result.docker_server is not None
    assert result.docker_server.start_command == f"%(ENV_{START_COMMAND_ENVVAR_NAME})s"
    assert (
        result.environment_variables[START_COMMAND_ENVVAR_NAME] == expected_vllm_command
    )


def test_render_truss_config_delegation():
    """Test that _render_truss_config_for_checkpoint_deployment delegates correctly based on model weight format."""
    deploy_config = DeployCheckpointsConfigComplete(
        checkpoint_details=definitions.CheckpointList(
            checkpoints=[
                definitions.LoRACheckpoint(
                    training_job_id="job123",
                    paths=["rank-0/checkpoint-1/"],
                    model_weight_format=definitions.ModelWeightsFormat.LORA,
                    lora_details=definitions.LoRADetails(rank=32),
                )
            ],
            base_model_id="google/gemma-3-27b-it",
        ),
        model_name="test-model",
        compute=definitions.Compute(
            accelerator=truss_config.AcceleratorSpec(accelerator="H100", count=4)
        ),
        runtime=definitions.DeployCheckpointsRuntime(environment_variables={}),
        deployment_name="test-deployment",
        model_weight_format=truss_config.ModelWeightsFormat.LORA,
    )

    # Test that it works for LoRA format
    result = _render_truss_config_for_checkpoint_deployment(deploy_config)
    assert isinstance(result, truss_config.TrussConfig)
    expected_vllm_command = "vllm serve google/gemma-3-27b-it --port 8000 --tensor-parallel-size 4 --enable-lora --max-lora-rank 32 --dtype bfloat16 --lora-modules job123=/tmp/training_checkpoints/job123/rank-0/checkpoint-1"
    assert (
        expected_vllm_command in result.environment_variables[START_COMMAND_ENVVAR_NAME]
    )
