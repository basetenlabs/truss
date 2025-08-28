import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

import truss_train.definitions as definitions
from truss.base import truss_config
from truss.cli.train.deploy_checkpoints import prepare_checkpoint_deploy
from truss.cli.train.deploy_checkpoints.deploy_checkpoints import (
    _get_checkpoint_ids_to_deploy,
    _render_truss_config_for_checkpoint_deployment,
    hydrate_checkpoint,
)
from truss.cli.train.deploy_checkpoints.deploy_full_checkpoints import (
    hydrate_full_checkpoint,
    render_vllm_full_truss_config,
)
from truss.cli.train.deploy_checkpoints.deploy_lora_checkpoints import (
    START_COMMAND_ENVVAR_NAME,
    _get_lora_rank,
    hydrate_lora_checkpoint,
    render_vllm_lora_truss_config,
)
from truss.cli.train.deploy_checkpoints.deploy_whisper_checkpoints import (
    VLLM_WHISPER_START_COMMAND,
    hydrate_whisper_checkpoint,
    render_vllm_whisper_truss_config,
)
from truss.cli.train.types import (
    DeployCheckpointsConfigComplete,
    PrepareCheckpointResult,
)
from truss_train.definitions import ModelWeightsFormat


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
def deploy_checkpoints_mock_input(create_mock_prompt):
    with patch(
        "truss.cli.train.deploy_checkpoints.deploy_checkpoints.inquirer.input"
    ) as mock:
        mock.side_effect = lambda message, **kwargs: create_mock_prompt(None)
        yield mock


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
            else "gemma-3-27b-it"
            if "model name" in message
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
                    model_weight_format=ModelWeightsFormat.LORA,
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
        model_weight_format=ModelWeightsFormat.LORA,
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
    deploy_checkpoints_mock_input,
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
                    model_weight_format=ModelWeightsFormat.LORA,
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
    assert checkpoint.model_weight_format == ModelWeightsFormat.LORA
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
            model_weight_format=ModelWeightsFormat.LORA,
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
                model_weight_format=ModelWeightsFormat.LORA,
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
                    model_weight_format=ModelWeightsFormat.LORA,
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
        model_weight_format=ModelWeightsFormat.LORA,
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
                    model_weight_format=ModelWeightsFormat.LORA,
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
        model_weight_format=ModelWeightsFormat.LORA,
    )

    # Test that it works for LoRA format
    result = _render_truss_config_for_checkpoint_deployment(deploy_config)
    assert isinstance(result, truss_config.TrussConfig)
    expected_vllm_command = "vllm serve google/gemma-3-27b-it --port 8000 --tensor-parallel-size 4 --enable-lora --max-lora-rank 32 --dtype bfloat16 --lora-modules job123=/tmp/training_checkpoints/job123/rank-0/checkpoint-1"
    assert (
        expected_vllm_command in result.environment_variables[START_COMMAND_ENVVAR_NAME]
    )


def test_render_vllm_full_truss_config():
    """Test that render_vllm_full_truss_config creates proper TrussConfig for full fine-tune deployments."""
    deploy_config = DeployCheckpointsConfigComplete(
        checkpoint_details=definitions.CheckpointList(
            checkpoints=[
                definitions.FullCheckpoint(
                    training_job_id="job123",
                    paths=["rank-0/checkpoint-1/"],
                    model_weight_format=ModelWeightsFormat.FULL,
                )
            ],
            base_model_id=None,  # Not needed for full fine-tune
        ),
        model_name="test-full-model",
        compute=definitions.Compute(
            accelerator=truss_config.AcceleratorSpec(accelerator="H100", count=2)
        ),
        runtime=definitions.DeployCheckpointsRuntime(
            environment_variables={
                "HF_TOKEN": definitions.SecretReference(name="hf_token")
            }
        ),
        deployment_name="test-deployment",
        model_weight_format=ModelWeightsFormat.FULL,
    )

    result = render_vllm_full_truss_config(deploy_config)
    expected_vllm_command = (
        "sh -c 'HF_TOKEN=$(cat /secrets/hf_token) "
        'HF_TOKEN="$$(cat /secrets/hf_access_token)" && export HF_TOKEN && '
        "if [ -f /tmp/training_checkpoints/job123/rank-0/checkpoint-1/chat_template.jinja ]; then   "
        "vllm serve /tmp/training_checkpoints/job123/rank-0/checkpoint-1 "
        "--chat-template /tmp/training_checkpoints/job123/rank-0/checkpoint-1/chat_template.jinja   "
        "--port 8000 --tensor-parallel-size 2 --dtype bfloat16; else   "
        "vllm serve /tmp/training_checkpoints/job123/rank-0/checkpoint-1 "
        "--port 8000 --tensor-parallel-size 2 --dtype bfloat16; fi'"
    )

    assert isinstance(result, truss_config.TrussConfig)
    assert result.model_name == "test-full-model"
    assert result.docker_server is not None
    assert result.docker_server.start_command == f"%(ENV_{START_COMMAND_ENVVAR_NAME})s"
    assert (
        result.environment_variables[START_COMMAND_ENVVAR_NAME] == expected_vllm_command
    )


def test_hydrate_full_checkpoint():
    """Test that hydrate_full_checkpoint creates proper FullCheckpoint objects."""
    job_id = "test_job_123"
    checkpoint_id = "checkpoint-456"
    checkpoint_data = {"base_model": "google/gemma-3-27b-it", "checkpoint_type": "full"}

    result = hydrate_full_checkpoint(job_id, checkpoint_id, checkpoint_data)

    assert isinstance(result, definitions.FullCheckpoint)
    assert result.training_job_id == job_id
    assert result.model_weight_format == ModelWeightsFormat.FULL
    assert len(result.paths) == 1
    assert result.paths[0] == f"rank-0/{checkpoint_id}/"


def test_hydrate_checkpoint_dispatcher_full():
    """Test that hydrate_checkpoint properly dispatches to full checkpoint function."""
    job_id = "test_job_123"
    checkpoint_id = "checkpoint-456"
    checkpoint_data = {"base_model": "google/gemma-3-27b-it", "checkpoint_type": "full"}

    result = hydrate_checkpoint(job_id, checkpoint_id, checkpoint_data, "full")
    assert isinstance(result, definitions.FullCheckpoint)
    assert result.model_weight_format == ModelWeightsFormat.FULL


def test_get_checkpoint_ids_to_deploy_full_checkpoints():
    """Test that _get_checkpoint_ids_to_deploy uses single selection for FULL checkpoints."""
    from collections import OrderedDict

    # Mock FULL checkpoints
    response_checkpoints = OrderedDict(
        [
            ("checkpoint-1", {"checkpoint_type": "full", "base_model": "test/model"}),
            ("checkpoint-2", {"checkpoint_type": "full", "base_model": "test/model"}),
            ("checkpoint-3", {"checkpoint_type": "full", "base_model": "test/model"}),
        ]
    )

    checkpoint_options = ["checkpoint-1", "checkpoint-2", "checkpoint-3"]

    with patch(
        "truss.cli.train.deploy_checkpoints.deploy_checkpoints.inquirer.checkbox"
    ) as mock_checkbox:
        mock_checkbox.return_value.execute.return_value = ["checkpoint-2"]

        result = _get_checkpoint_ids_to_deploy(checkpoint_options, response_checkpoints)

        # Should use checkbox (single selection) for FULL checkpoints
        mock_checkbox.assert_called_once()
        assert (
            mock_checkbox.call_args[1]["message"]
            == "Use spacebar to select/deselect checkpoints to deploy. Press enter when done."
        )
        assert mock_checkbox.call_args[1]["choices"] == checkpoint_options

        # Should return a list with single selected checkpoint
        assert result == ["checkpoint-2"]


def test_get_checkpoint_ids_to_deploy_lora_checkpoints():
    """Test that _get_checkpoint_ids_to_deploy uses multiple selection for LoRA checkpoints."""
    from collections import OrderedDict

    # Mock LoRA checkpoints
    response_checkpoints = OrderedDict(
        [
            ("checkpoint-1", {"checkpoint_type": "lora", "base_model": "test/model"}),
            ("checkpoint-2", {"checkpoint_type": "lora", "base_model": "test/model"}),
            ("checkpoint-3", {"checkpoint_type": "lora", "base_model": "test/model"}),
        ]
    )

    checkpoint_options = ["checkpoint-1", "checkpoint-2", "checkpoint-3"]

    with patch(
        "truss.cli.train.deploy_checkpoints.deploy_checkpoints.inquirer.checkbox"
    ) as mock_checkbox:
        mock_checkbox.return_value.execute.return_value = [
            "checkpoint-1",
            "checkpoint-3",
        ]

        result = _get_checkpoint_ids_to_deploy(checkpoint_options, response_checkpoints)

        # Should use checkbox (multiple selection) for LoRA checkpoints
        mock_checkbox.assert_called_once()
        assert (
            mock_checkbox.call_args[1]["message"]
            == "Use spacebar to select/deselect checkpoints to deploy. Press enter when done."
        )
        assert mock_checkbox.call_args[1]["choices"] == checkpoint_options

        # Should return a list with multiple selected checkpoints
        assert result == ["checkpoint-1", "checkpoint-3"]


def test_get_checkpoint_ids_to_deploy_mixed_checkpoints():
    """Test that _get_checkpoint_ids_to_deploy uses multiple selection for mixed checkpoint types."""
    from collections import OrderedDict

    # Mock mixed checkpoints (FULL and LoRA)
    response_checkpoints = OrderedDict(
        [
            ("checkpoint-1", {"checkpoint_type": "full", "base_model": "test/model"}),
            ("checkpoint-2", {"checkpoint_type": "lora", "base_model": "test/model"}),
            ("checkpoint-3", {"checkpoint_type": "full", "base_model": "test/model"}),
        ]
    )

    checkpoint_options = ["checkpoint-1", "checkpoint-2", "checkpoint-3"]

    with patch(
        "truss.cli.train.deploy_checkpoints.deploy_checkpoints.inquirer.checkbox"
    ) as mock_checkbox:
        # For mixed checkpoints with FULL, we can only select one
        mock_checkbox.return_value.execute.return_value = ["checkpoint-1"]

        result = _get_checkpoint_ids_to_deploy(checkpoint_options, response_checkpoints)

        # Should use checkbox (multiple selection) for mixed checkpoint types
        mock_checkbox.assert_called_once()
        assert (
            mock_checkbox.call_args[1]["message"]
            == "Use spacebar to select/deselect checkpoints to deploy. Press enter when done."
        )
        assert mock_checkbox.call_args[1]["choices"] == checkpoint_options

        # Should return a list with single selected checkpoint (due to FULL checkpoint)
        assert result == ["checkpoint-1"]


def test_get_checkpoint_ids_to_deploy_single_checkpoint():
    """Test that _get_checkpoint_ids_to_deploy returns single checkpoint without prompting."""
    from collections import OrderedDict

    # Mock single checkpoint
    response_checkpoints = OrderedDict(
        [("checkpoint-1", {"checkpoint_type": "full", "base_model": "test/model"})]
    )

    checkpoint_options = ["checkpoint-1"]

    # Should not call any inquirer functions for single checkpoint
    with patch(
        "truss.cli.train.deploy_checkpoints.deploy_checkpoints.inquirer.select"
    ) as mock_select, patch(
        "truss.cli.train.deploy_checkpoints.deploy_checkpoints.inquirer.checkbox"
    ) as mock_checkbox:
        result = _get_checkpoint_ids_to_deploy(checkpoint_options, response_checkpoints)

        # Should not call any inquirer functions
        mock_select.assert_not_called()
        mock_checkbox.assert_not_called()

        # Should return the single checkpoint directly
        assert result == ["checkpoint-1"]


def test_vllm_whisper_start_command_template():
    """Test that the VLLM_WHISPER_START_COMMAND template renders correctly."""
    # Test with all variables
    result = VLLM_WHISPER_START_COMMAND.render(
        model_path="/path/to/model",
        envvars="CUDA_VISIBLE_DEVICES=0",
        specify_tensor_parallelism=4,
    )

    expected = (
        "sh -c 'CUDA_VISIBLE_DEVICES=0 "
        'HF_TOKEN="$$(cat /secrets/hf_access_token)" && export HF_TOKEN && '
        "vllm serve /path/to/model --port 8000 --tensor-parallel-size 4'"
    )
    assert result == expected

    result = VLLM_WHISPER_START_COMMAND.render(
        model_path="/path/to/model", envvars=None, specify_tensor_parallelism=1
    )

    expected = (
        "sh -c '"
        'HF_TOKEN="$$(cat /secrets/hf_access_token)" && export HF_TOKEN && '
        "vllm serve /path/to/model --port 8000 --tensor-parallel-size 1'"
    )
    assert result == expected


def test_hydrate_whisper_checkpoint():
    """Test that hydrate_whisper_checkpoint creates correct WhisperCheckpoint object."""
    job_id = "test-job-123"
    checkpoint_id = "checkpoint-456"
    checkpoint = {"some": "data"}

    result = hydrate_whisper_checkpoint(job_id, checkpoint_id, checkpoint)

    assert result.training_job_id == job_id
    assert result.paths == [f"rank-0/{checkpoint_id}/"]
    assert result.model_weight_format == definitions.ModelWeightsFormat.WHISPER
    assert isinstance(result, definitions.WhisperCheckpoint)


@patch(
    "truss.cli.train.deploy_checkpoints.deploy_whisper_checkpoints.setup_base_truss_config"
)
@patch(
    "truss.cli.train.deploy_checkpoints.deploy_whisper_checkpoints.setup_environment_variables_and_secrets"
)
@patch(
    "truss.cli.train.deploy_checkpoints.deploy_whisper_checkpoints.build_full_checkpoint_string"
)
def test_render_vllm_whisper_truss_config(
    mock_build_full_checkpoint_string, mock_setup_env_vars, mock_setup_base_config
):
    """Test that render_vllm_whisper_truss_config renders truss config correctly."""
    # Mock dependencies
    mock_truss_config = MagicMock()
    mock_truss_config.environment_variables = {}
    mock_truss_config.docker_server = MagicMock()
    mock_setup_base_config.return_value = mock_truss_config

    mock_setup_env_vars.return_value = "HF_TOKEN=$(cat /secrets/hf_access_token)"
    mock_build_full_checkpoint_string.return_value = "/path/to/checkpoint"

    # Create test config
    deploy_config = DeployCheckpointsConfigComplete(
        checkpoint_details=definitions.CheckpointList(
            checkpoints=[
                definitions.WhisperCheckpoint(
                    training_job_id="job123",
                    paths=["rank-0/checkpoint-1/"],
                    model_weight_format=definitions.ModelWeightsFormat.WHISPER,
                )
            ],
            base_model_id="openai/whisper-large-v3",
        ),
        model_name="whisper-large-v3-vLLM",
        compute=definitions.Compute(
            accelerator=truss_config.AcceleratorSpec(accelerator="H100", count=4)
        ),
        runtime=definitions.DeployCheckpointsRuntime(
            environment_variables={
                "HF_TOKEN": definitions.SecretReference(name="hf_access_token")
            }
        ),
        deployment_name="whisper-large-v3-vLLM",
        model_weight_format=definitions.ModelWeightsFormat.WHISPER,
    )

    result = render_vllm_whisper_truss_config(deploy_config)

    mock_setup_base_config.assert_called_once_with(deploy_config)
    mock_setup_env_vars.assert_called_once_with(mock_truss_config, deploy_config)
    mock_build_full_checkpoint_string.assert_called_once_with(mock_truss_config)

    assert result == mock_truss_config

    expected_start_command = (
        "sh -c 'HF_TOKEN=$(cat /secrets/hf_access_token) "
        'HF_TOKEN="$$(cat /secrets/hf_access_token)" && export HF_TOKEN && '
        "vllm serve /path/to/checkpoint --port 8000 --tensor-parallel-size 4'"
    )
    assert (
        result.environment_variables[START_COMMAND_ENVVAR_NAME]
        == expected_start_command
    )

    assert result.docker_server.start_command == f"%(ENV_{START_COMMAND_ENVVAR_NAME})s"


@patch(
    "truss.cli.train.deploy_checkpoints.deploy_whisper_checkpoints.setup_base_truss_config"
)
@patch(
    "truss.cli.train.deploy_checkpoints.deploy_whisper_checkpoints.setup_environment_variables_and_secrets"
)
@patch(
    "truss.cli.train.deploy_checkpoints.deploy_whisper_checkpoints.build_full_checkpoint_string"
)
def test_render_vllm_whisper_truss_config_with_envvars(
    mock_build_full_checkpoint_string, mock_setup_env_vars, mock_setup_base_config
):
    """Test that render_vllm_whisper_truss_config handles environment variables correctly."""
    # Mock dependencies
    mock_truss_config = MagicMock()
    mock_truss_config.environment_variables = {}
    mock_truss_config.docker_server = MagicMock()
    mock_setup_base_config.return_value = mock_truss_config

    mock_setup_env_vars.return_value = "CUDA_VISIBLE_DEVICES=0,1"
    mock_build_full_checkpoint_string.return_value = "/path/to/checkpoint"

    # Create test config with environment variables
    deploy_config = DeployCheckpointsConfigComplete(
        checkpoint_details=definitions.CheckpointList(
            checkpoints=[
                definitions.WhisperCheckpoint(
                    training_job_id="job123",
                    paths=["rank-0/checkpoint-1/"],
                    model_weight_format=definitions.ModelWeightsFormat.WHISPER,
                )
            ],
            base_model_id="openai/whisper-large-v3",
        ),
        model_name="whisper-large-v3-vLLM",
        compute=definitions.Compute(
            accelerator=truss_config.AcceleratorSpec(accelerator="H100", count=2)
        ),
        runtime=definitions.DeployCheckpointsRuntime(
            environment_variables={
                "CUDA_VISIBLE_DEVICES": "0,1",
                "HF_TOKEN": definitions.SecretReference(name="hf_access_token"),
            }
        ),
        deployment_name="whisper-large-v3-vLLM",
        model_weight_format=definitions.ModelWeightsFormat.WHISPER,
    )

    # Call function under test
    result = render_vllm_whisper_truss_config(deploy_config)

    # Verify environment variables are included in start command
    expected_start_command = (
        "sh -c 'CUDA_VISIBLE_DEVICES=0,1 "
        'HF_TOKEN="$$(cat /secrets/hf_access_token)" && export HF_TOKEN && '
        "vllm serve /path/to/checkpoint --port 8000 --tensor-parallel-size 2'"
    )
    assert (
        result.environment_variables[START_COMMAND_ENVVAR_NAME]
        == expected_start_command
    )


@dataclass
class TestCase:
    """Test case for setup_base_truss_config function."""

    desc: str
    input_config: DeployCheckpointsConfigComplete
    expected_model_name: str
    expected_predict_endpoint: str
    expected_accelerator: Optional[str]
    expected_accelerator_count: Optional[int]
    expected_checkpoint_paths: List[str]
    expected_environment_variables: Dict[str, str]
    should_raise: Optional[str] = None  # Error message if function should raise

    __test__ = False  # Tell pytest this is not a test class


def test_setup_base_truss_config():
    """Table-driven test for setup_base_truss_config function."""
    from truss.cli.train.deploy_checkpoints.deploy_checkpoints_helpers import (
        setup_base_truss_config,
    )

    # Define test cases
    test_cases = [
        TestCase(
            desc="LoRA checkpoint with H100 accelerator",
            input_config=DeployCheckpointsConfigComplete(
                checkpoint_details=definitions.CheckpointList(
                    checkpoints=[
                        definitions.LoRACheckpoint(
                            training_job_id="job123",
                            paths=["rank-0/checkpoint-1/"],
                            model_weight_format=ModelWeightsFormat.LORA,
                            lora_details=definitions.LoRADetails(rank=32),
                        )
                    ],
                    base_model_id="google/gemma-3-27b-it",
                ),
                model_name="test-lora-model",
                compute=definitions.Compute(
                    accelerator=truss_config.AcceleratorSpec(
                        accelerator="H100", count=4
                    )
                ),
                runtime=definitions.DeployCheckpointsRuntime(environment_variables={}),
                deployment_name="test-deployment",
                model_weight_format=ModelWeightsFormat.LORA,
            ),
            expected_model_name="test-lora-model",
            expected_predict_endpoint="/v1/chat/completions",
            expected_accelerator="H100",
            expected_accelerator_count=4,
            expected_checkpoint_paths=["rank-0/checkpoint-1/"],
            expected_environment_variables={
                "VLLM_LOGGING_LEVEL": "WARNING",
                "VLLM_USE_V1": "0",
                "HF_HUB_ENABLE_HF_TRANSFER": "1",
            },
        ),
        TestCase(
            desc="Whisper checkpoint with A100 accelerator",
            input_config=DeployCheckpointsConfigComplete(
                checkpoint_details=definitions.CheckpointList(
                    checkpoints=[
                        definitions.WhisperCheckpoint(
                            training_job_id="job123",
                            paths=["rank-0/checkpoint-1/"],
                            model_weight_format=definitions.ModelWeightsFormat.WHISPER,
                        )
                    ],
                    base_model_id="openai/whisper-large-v3",
                ),
                model_name="test-whisper-model",
                compute=definitions.Compute(
                    accelerator=truss_config.AcceleratorSpec(
                        accelerator="A100", count=2
                    )
                ),
                runtime=definitions.DeployCheckpointsRuntime(environment_variables={}),
                deployment_name="test-whisper-deployment",
                model_weight_format=definitions.ModelWeightsFormat.WHISPER,
            ),
            expected_model_name="test-whisper-model",
            expected_predict_endpoint="/v1/audio/transcriptions",
            expected_accelerator="A100",
            expected_accelerator_count=2,
            expected_checkpoint_paths=["rank-0/checkpoint-1/"],
            expected_environment_variables={
                "VLLM_LOGGING_LEVEL": "WARNING",
                "VLLM_USE_V1": "0",
                "HF_HUB_ENABLE_HF_TRANSFER": "1",
            },
        ),
        TestCase(
            desc="Multiple LoRA checkpoints",
            input_config=DeployCheckpointsConfigComplete(
                checkpoint_details=definitions.CheckpointList(
                    checkpoints=[
                        definitions.LoRACheckpoint(
                            training_job_id="job123",
                            paths=["rank-0/checkpoint-1/"],
                            model_weight_format=ModelWeightsFormat.LORA,
                            lora_details=definitions.LoRADetails(rank=16),
                        ),
                        definitions.LoRACheckpoint(
                            training_job_id="job123",
                            paths=["rank-0/checkpoint-2/"],
                            model_weight_format=ModelWeightsFormat.LORA,
                            lora_details=definitions.LoRADetails(rank=32),
                        ),
                    ],
                    base_model_id="google/gemma-3-27b-it",
                ),
                model_name="test-multi-checkpoint-model",
                compute=definitions.Compute(
                    accelerator=truss_config.AcceleratorSpec(
                        accelerator="H100", count=4
                    )
                ),
                runtime=definitions.DeployCheckpointsRuntime(environment_variables={}),
                deployment_name="test-multi-deployment",
                model_weight_format=ModelWeightsFormat.LORA,
            ),
            expected_model_name="test-multi-checkpoint-model",
            expected_predict_endpoint="/v1/chat/completions",
            expected_accelerator="H100",
            expected_accelerator_count=4,
            expected_checkpoint_paths=["rank-0/checkpoint-1/", "rank-0/checkpoint-2/"],
            expected_environment_variables={
                "VLLM_LOGGING_LEVEL": "WARNING",
                "VLLM_USE_V1": "0",
                "HF_HUB_ENABLE_HF_TRANSFER": "1",
            },
        ),
        TestCase(
            desc="No accelerator specified",
            input_config=DeployCheckpointsConfigComplete(
                checkpoint_details=definitions.CheckpointList(
                    checkpoints=[
                        definitions.LoRACheckpoint(
                            training_job_id="job123",
                            paths=["rank-0/checkpoint-1/"],
                            model_weight_format=ModelWeightsFormat.LORA,
                            lora_details=definitions.LoRADetails(rank=16),
                        )
                    ],
                    base_model_id="google/gemma-3-27b-it",
                ),
                model_name="test-no-accelerator-model",
                compute=definitions.Compute(),  # No accelerator specified
                runtime=definitions.DeployCheckpointsRuntime(environment_variables={}),
                deployment_name="test-no-accelerator-deployment",
                model_weight_format=ModelWeightsFormat.LORA,
            ),
            expected_model_name="test-no-accelerator-model",
            expected_predict_endpoint="/v1/chat/completions",
            expected_accelerator=None,
            expected_accelerator_count=None,
            expected_checkpoint_paths=["rank-0/checkpoint-1/"],
            expected_environment_variables={
                "VLLM_LOGGING_LEVEL": "WARNING",
                "VLLM_USE_V1": "0",
                "HF_HUB_ENABLE_HF_TRANSFER": "1",
            },
        ),
    ]

    # Run test cases
    for test_case in test_cases:
        print(f"Running test case: {test_case.desc}")

        if test_case.should_raise:
            # Test error cases
            with pytest.raises(Exception, match=test_case.should_raise):
                setup_base_truss_config(test_case.input_config)
        else:
            # Test success cases
            result = setup_base_truss_config(test_case.input_config)

            # Verify basic structure
            assert isinstance(result, truss_config.TrussConfig), (
                f"Test case '{test_case.desc}': Result should be TrussConfig"
            )
            assert result.model_name == test_case.expected_model_name, (
                f"Test case '{test_case.desc}': Model name mismatch"
            )

            # Verify docker server configuration
            assert result.docker_server is not None, (
                f"Test case '{test_case.desc}': Docker server should not be None"
            )
            assert result.docker_server.start_command == 'sh -c ""', (
                f"Test case '{test_case.desc}': Start command mismatch"
            )
            assert result.docker_server.readiness_endpoint == "/health", (
                f"Test case '{test_case.desc}': Readiness endpoint mismatch"
            )
            assert result.docker_server.liveness_endpoint == "/health", (
                f"Test case '{test_case.desc}': Liveness endpoint mismatch"
            )
            assert (
                result.docker_server.predict_endpoint
                == test_case.expected_predict_endpoint
            ), f"Test case '{test_case.desc}': Predict endpoint mismatch"
            assert result.docker_server.server_port == 8000, (
                f"Test case '{test_case.desc}': Server port mismatch"
            )

            # Verify training checkpoints
            assert result.training_checkpoints is not None, (
                f"Test case '{test_case.desc}': Training checkpoints should not be None"
            )
            assert len(result.training_checkpoints.artifact_references) == len(
                test_case.expected_checkpoint_paths
            ), f"Test case '{test_case.desc}': Number of checkpoint artifacts mismatch"

            for i, expected_path in enumerate(test_case.expected_checkpoint_paths):
                artifact_ref = result.training_checkpoints.artifact_references[i]
                assert artifact_ref.paths == [expected_path], (
                    f"Test case '{test_case.desc}': Checkpoint path {i} mismatch"
                )

            # Verify resources
            assert result.resources is not None, (
                f"Test case '{test_case.desc}': Resources should not be None"
            )

            if test_case.expected_accelerator:
                assert result.resources.accelerator is not None, (
                    f"Test case '{test_case.desc}': Accelerator should not be None"
                )
                assert (
                    result.resources.accelerator.accelerator
                    == test_case.expected_accelerator
                ), f"Test case '{test_case.desc}': Accelerator type mismatch"
                assert (
                    result.resources.accelerator.count
                    == test_case.expected_accelerator_count
                ), f"Test case '{test_case.desc}': Accelerator count mismatch"
            else:
                # When no accelerator is specified, it creates an AcceleratorSpec with None values
                assert result.resources.accelerator is not None, (
                    f"Test case '{test_case.desc}': Accelerator should exist"
                )
                assert result.resources.accelerator.accelerator is None, (
                    f"Test case '{test_case.desc}': Accelerator type should be None"
                )

            # Verify environment variables
            for key, expected_value in test_case.expected_environment_variables.items():
                assert result.environment_variables[key] == expected_value, (
                    f"Test case '{test_case.desc}': Environment variable {key} mismatch"
                )
