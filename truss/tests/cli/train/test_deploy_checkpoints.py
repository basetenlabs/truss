from unittest.mock import MagicMock, patch

import pytest
import rich_click as click

import truss_train.definitions as definitions
from truss.cli.train.deploy_checkpoints.deploy_checkpoints import (
    _ensure_trainer_checkpoint_details,
    _get_checkpoint_ids_to_deploy,
    _resolve_trainer,
    hydrate_checkpoint,
)
from truss.cli.train.deploy_checkpoints.deploy_full_checkpoints import (
    hydrate_full_checkpoint,
)
from truss.cli.train.deploy_checkpoints.deploy_whisper_checkpoints import (
    hydrate_whisper_checkpoint,
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


def test_hydrate_full_checkpoint():
    """Test that hydrate_full_checkpoint creates proper FullCheckpoint objects."""
    job_id = "test_job_123"
    checkpoint_id = "checkpoint-456"
    checkpoint_data = {"base_model": "google/gemma-3-27b-it", "checkpoint_type": "full"}

    result = hydrate_full_checkpoint(job_id, checkpoint_id, checkpoint_data)

    assert isinstance(result, definitions.FullCheckpoint)
    assert result.training_job_id == job_id
    assert result.model_weight_format == ModelWeightsFormat.FULL
    assert result.checkpoint_name == checkpoint_id


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
    with (
        patch(
            "truss.cli.train.deploy_checkpoints.deploy_checkpoints.inquirer.select"
        ) as mock_select,
        patch(
            "truss.cli.train.deploy_checkpoints.deploy_checkpoints.inquirer.checkbox"
        ) as mock_checkbox,
    ):
        result = _get_checkpoint_ids_to_deploy(checkpoint_options, response_checkpoints)

        # Should not call any inquirer functions
        mock_select.assert_not_called()
        mock_checkbox.assert_not_called()

        # Should return the single checkpoint directly
        assert result == ["checkpoint-1"]


def test_hydrate_whisper_checkpoint():
    """Test that hydrate_whisper_checkpoint creates correct WhisperCheckpoint object."""
    job_id = "test-job-123"
    checkpoint_id = "checkpoint-456"
    checkpoint = {"some": "data"}

    result = hydrate_whisper_checkpoint(job_id, checkpoint_id, checkpoint)

    assert result.training_job_id == job_id
    assert result.checkpoint_name == checkpoint_id
    assert result.model_weight_format == definitions.ModelWeightsFormat.WHISPER
    assert isinstance(result, definitions.WhisperCheckpoint)


@pytest.fixture
def mock_trainer_remote():
    mock = MagicMock()
    mock.api.search_trainers.return_value = [
        {"trainer_id": "trnr_xyz", "session_id": "sess_abc", "base_model": "Qwen/Qwen3-8B"}
    ]
    mock.api.list_trainer_checkpoints.return_value = {
        "trainer_id": "trnr_xyz",
        "checkpoints": [
            {
                "id": "tcp_step100",
                "trainer_id": "trnr_xyz",
                "checkpoint_id": "step-100",
                "base_model": "Qwen/Qwen3-8B",
                "checkpoint_type": "lora",
                "lora_adapter_config": {"r": 16},
            }
        ],
    }
    return mock


def test_resolve_trainer_returns_first_match(mock_trainer_remote):
    result = _resolve_trainer(mock_trainer_remote, "trnr_xyz")
    assert result["trainer_id"] == "trnr_xyz"
    assert result["session_id"] == "sess_abc"
    mock_trainer_remote.api.search_trainers.assert_called_once_with(trainer_id="trnr_xyz")


def test_resolve_trainer_raises_when_not_found(mock_trainer_remote):
    mock_trainer_remote.api.search_trainers.return_value = []
    with pytest.raises(click.UsageError, match="Trainer trnr_missing not found"):
        _resolve_trainer(mock_trainer_remote, "trnr_missing")


def test_ensure_trainer_checkpoint_details_user_provided_passes_through(
    mock_trainer_remote,
):
    """When the user authored trainer_checkpoint_ids in --config, return as-is."""
    user_config = definitions.CheckpointList(trainer_checkpoint_ids=["tcp_step100"])
    result = _ensure_trainer_checkpoint_details(
        mock_trainer_remote, user_config, trainer_id=None
    )
    assert result is user_config
    # Did not hit the API since user authored the IDs.
    mock_trainer_remote.api.search_trainers.assert_not_called()
    mock_trainer_remote.api.list_trainer_checkpoints.assert_not_called()


def test_ensure_trainer_checkpoint_details_requires_trainer_id_when_unprovided(
    mock_trainer_remote,
):
    with pytest.raises(click.UsageError, match="--trainer-id is required"):
        _ensure_trainer_checkpoint_details(
            mock_trainer_remote, checkpoint_details=None, trainer_id=None
        )


def test_ensure_trainer_checkpoint_details_picker_emits_ids_and_base_model(
    mock_trainer_remote,
):
    """With --trainer-id set, picker selects checkpoints and we send IDs on the wire.

    Single checkpoint short-circuits the inquirer prompt (mirroring TJC behavior),
    so no prompt mock is needed for this case.
    """
    result = _ensure_trainer_checkpoint_details(
        mock_trainer_remote,
        checkpoint_details=None,
        trainer_id="trnr_xyz",
    )
    assert result.trainer_checkpoint_ids == ["tcp_step100"]
    assert result.base_model_id == "Qwen/Qwen3-8B"
    mock_trainer_remote.api.list_trainer_checkpoints.assert_called_once_with(
        "sess_abc", "trnr_xyz"
    )
