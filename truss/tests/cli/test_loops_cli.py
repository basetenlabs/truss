"""Tests for truss loops CLI commands."""

import os
from unittest.mock import Mock, patch

import pytest
from click.testing import CliRunner

from truss.cli.cli import truss_cli


@pytest.fixture
def mock_remote():
    remote = Mock()
    remote.create_trainer_session.return_value = {"id": "session_abc123"}
    remote.create_trainer_server.return_value = {
        "id": "trainer_xyz456",
        "base_url": "https://trainer-xyz456.api.baseten.co/trainer",
        "sampling_server": {
            "id": "sampler_def789",
            "base_url": "https://model-def789.api.baseten.co/deployment/v1/sync",
        },
    }
    return remote


def _invoke_loops_push(args, mock_remote):
    runner = CliRunner()
    with patch(
        "truss.remote.remote_factory.RemoteFactory.create", return_value=mock_remote
    ):
        return runner.invoke(truss_cli, ["loops", "push"] + args)


def test_push_basic(mock_remote):
    result = _invoke_loops_push(
        ["Qwen/Qwen3-8B", "--remote", "test_remote"], mock_remote
    )

    assert result.exit_code == 0, result.output
    mock_remote.create_trainer_session.assert_called_once_with(training_project_id=None)
    mock_remote.create_trainer_server.assert_called_once_with(
        session_id="session_abc123",
        model="Qwen/Qwen3-8B",
        max_seq_len=4096,
        sampler_checkpoint_id=None,
        trainer_checkpoint_id=None,
    )
    assert "Qwen/Qwen3-8B" in result.output


def test_push_with_training_project_id(mock_remote):
    result = _invoke_loops_push(
        [
            "Qwen/Qwen3-8B",
            "--remote",
            "test_remote",
            "--training-project-id",
            "proj_abc",
        ],
        mock_remote,
    )

    assert result.exit_code == 0, result.output
    mock_remote.create_trainer_session.assert_called_once_with(
        training_project_id="proj_abc"
    )


def test_push_with_sampler_checkpoint(mock_remote):
    result = _invoke_loops_push(
        [
            "Qwen/Qwen3-8B",
            "--remote",
            "test_remote",
            "--sampler-checkpoint",
            "ckpt_sampler",
        ],
        mock_remote,
    )

    assert result.exit_code == 0, result.output
    mock_remote.create_trainer_server.assert_called_once_with(
        session_id="session_abc123",
        model="Qwen/Qwen3-8B",
        max_seq_len=4096,
        sampler_checkpoint_id="ckpt_sampler",
        trainer_checkpoint_id=None,
    )


def test_push_with_trainer_checkpoint(mock_remote):
    result = _invoke_loops_push(
        [
            "Qwen/Qwen3-8B",
            "--remote",
            "test_remote",
            "--trainer-checkpoint",
            "ckpt_trainer",
        ],
        mock_remote,
    )

    assert result.exit_code == 0, result.output
    mock_remote.create_trainer_server.assert_called_once_with(
        session_id="session_abc123",
        model="Qwen/Qwen3-8B",
        max_seq_len=4096,
        sampler_checkpoint_id=None,
        trainer_checkpoint_id="ckpt_trainer",
    )


def test_push_with_all_options(mock_remote):
    result = _invoke_loops_push(
        [
            "Qwen/Qwen3-8B",
            "--remote",
            "test_remote",
            "--training-project-id",
            "proj_abc",
            "--sampler-checkpoint",
            "ckpt_sampler",
            "--trainer-checkpoint",
            "ckpt_trainer",
        ],
        mock_remote,
    )

    assert result.exit_code == 0, result.output
    mock_remote.create_trainer_session.assert_called_once_with(
        training_project_id="proj_abc"
    )
    mock_remote.create_trainer_server.assert_called_once_with(
        session_id="session_abc123",
        model="Qwen/Qwen3-8B",
        max_seq_len=4096,
        sampler_checkpoint_id="ckpt_sampler",
        trainer_checkpoint_id="ckpt_trainer",
    )


def test_push_with_max_seq_len(mock_remote):
    result = _invoke_loops_push(
        ["Qwen/Qwen3-8B", "--remote", "test_remote", "--max-seq-len", "32768"],
        mock_remote,
    )

    assert result.exit_code == 0, result.output
    mock_remote.create_trainer_server.assert_called_once_with(
        session_id="session_abc123",
        model="Qwen/Qwen3-8B",
        max_seq_len=32768,
        sampler_checkpoint_id=None,
        trainer_checkpoint_id=None,
    )


def test_push_uses_inquire_when_remote_not_provided(mock_remote):
    runner = CliRunner()
    with patch(
        "truss.remote.remote_factory.RemoteFactory.create", return_value=mock_remote
    ):
        with patch(
            "truss.cli.remote_cli.inquire_remote_name", return_value="inquired_remote"
        ) as mock_inquire:
            result = runner.invoke(truss_cli, ["loops", "push", "Qwen/Qwen3-8B"])

    assert result.exit_code == 0, result.output
    mock_inquire.assert_called_once()


def test_push_fails_when_base_model_id_missing():
    runner = CliRunner()
    result = runner.invoke(truss_cli, ["loops", "push", "--remote", "test_remote"])

    assert result.exit_code != 0
    assert "Missing argument" in result.output


def test_push_propagates_session_creation_error(mock_remote):
    mock_remote.create_trainer_session.side_effect = RuntimeError(
        "session creation failed"
    )

    result = _invoke_loops_push(
        ["Qwen/Qwen3-8B", "--remote", "test_remote"], mock_remote
    )

    assert result.exit_code != 0


def test_push_propagates_trainer_server_creation_error(mock_remote):
    mock_remote.create_trainer_server.side_effect = RuntimeError(
        "active TrainerDeployment already exists"
    )

    result = _invoke_loops_push(
        ["Qwen/Qwen3-8B", "--remote", "test_remote"], mock_remote
    )

    assert result.exit_code != 0


def test_push_help():
    env = os.environ.copy()
    env["COLUMNS"] = "200"
    runner = CliRunner(env=env)
    result = runner.invoke(truss_cli, ["loops", "push", "--help"])

    assert result.exit_code == 0
    assert "--training-project-id" in result.output
    assert "--sampler-checkpoint" in result.output
    assert "--trainer-checkpoint" in result.output
    assert "--max-seq-len" in result.output
