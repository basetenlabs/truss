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
    remote.fetch_auth_header.return_value = {"Authorization": "Api-Key test_key"}
    return remote


def _invoke_loops_push(args, mock_remote):
    runner = CliRunner()
    with patch(
        "truss.remote.remote_factory.RemoteFactory.create", return_value=mock_remote
    ):
        with patch("requests.get") as mock_get:
            mock_get.return_value = Mock(status_code=200)
            return runner.invoke(truss_cli, ["loops", "push"] + args)


def test_push_basic(mock_remote):
    result = _invoke_loops_push(
        ["Qwen/Qwen3-8B", "--remote", "test_remote"], mock_remote
    )

    assert result.exit_code == 0, result.output
    mock_remote.create_trainer_session.assert_called_once_with(training_project_id=None)
    mock_remote.create_trainer_server.assert_called_once_with(
        session_id="session_abc123", model="Qwen/Qwen3-8B", max_seq_len=4096
    )
    assert "Qwen/Qwen3-8B" in result.output


def test_push_with_project_id(mock_remote):
    result = _invoke_loops_push(
        ["Qwen/Qwen3-8B", "--remote", "test_remote", "--project-id", "proj_abc"],
        mock_remote,
    )

    assert result.exit_code == 0, result.output
    mock_remote.create_trainer_session.assert_called_once_with(
        training_project_id="proj_abc"
    )


def test_push_with_max_seq_len(mock_remote):
    result = _invoke_loops_push(
        ["Qwen/Qwen3-8B", "--remote", "test_remote", "--max-seq-len", "32768"],
        mock_remote,
    )

    assert result.exit_code == 0, result.output
    mock_remote.create_trainer_server.assert_called_once_with(
        session_id="session_abc123", model="Qwen/Qwen3-8B", max_seq_len=32768
    )


def test_push_polls_until_running(mock_remote):
    # First two polls return 503, third returns 200.
    responses = [Mock(status_code=503), Mock(status_code=503), Mock(status_code=200)]

    runner = CliRunner()
    with patch(
        "truss.remote.remote_factory.RemoteFactory.create", return_value=mock_remote
    ):
        with patch("requests.get", side_effect=responses):
            with patch("truss.cli.loops_commands.time.sleep"):
                result = runner.invoke(
                    truss_cli,
                    ["loops", "push", "Qwen/Qwen3-8B", "--remote", "test_remote"],
                )

    assert result.exit_code == 0, result.output


def test_push_times_out_waiting_for_health(mock_remote):
    runner = CliRunner()
    with patch(
        "truss.remote.remote_factory.RemoteFactory.create", return_value=mock_remote
    ):
        with patch("requests.get", return_value=Mock(status_code=503)):
            with patch("truss.cli.loops_commands.time.sleep"):
                with patch(
                    "truss.cli.loops_commands.time.monotonic", side_effect=[0, 0, 700]
                ):
                    result = runner.invoke(
                        truss_cli,
                        ["loops", "push", "Qwen/Qwen3-8B", "--remote", "test_remote"],
                    )

    assert result.exit_code != 0
    assert "Timed out" in result.output


def test_push_uses_inquire_when_remote_not_provided(mock_remote):
    runner = CliRunner()
    with patch(
        "truss.remote.remote_factory.RemoteFactory.create", return_value=mock_remote
    ):
        with patch(
            "truss.cli.remote_cli.inquire_remote_name", return_value="inquired_remote"
        ) as mock_inquire:
            with patch("requests.get") as mock_get:
                mock_get.return_value = Mock(status_code=200)
                result = runner.invoke(truss_cli, ["loops", "push", "Qwen/Qwen3-8B"])

    assert result.exit_code == 0, result.output
    mock_inquire.assert_called_once()


def test_push_fails_when_base_model_missing():
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
    assert "--project-id" in result.output
    assert "--max-seq-len" in result.output


def _invoke_loops_deactivate(args, mock_remote, input=None):
    runner = CliRunner()
    with patch(
        "truss.remote.remote_factory.RemoteFactory.create", return_value=mock_remote
    ):
        return runner.invoke(
            truss_cli, ["loops", "deactivate"] + args, input=input
        )


def test_deactivate_basic(mock_remote):
    result = _invoke_loops_deactivate(
        ["--remote", "test_remote", "--yes"], mock_remote
    )

    assert result.exit_code == 0, result.output
    mock_remote.deactivate_trainer_deployment.assert_called_once()
    assert "deactivated" in result.output


def test_deactivate_confirms_before_proceeding(mock_remote):
    result = _invoke_loops_deactivate(
        ["--remote", "test_remote"], mock_remote, input="y\n"
    )

    assert result.exit_code == 0, result.output
    mock_remote.deactivate_trainer_deployment.assert_called_once()


def test_deactivate_aborts_on_no_confirmation(mock_remote):
    result = _invoke_loops_deactivate(
        ["--remote", "test_remote"], mock_remote, input="n\n"
    )

    assert result.exit_code != 0
    mock_remote.deactivate_trainer_deployment.assert_not_called()


def test_deactivate_uses_inquire_when_remote_not_provided(mock_remote):
    runner = CliRunner()
    with patch(
        "truss.remote.remote_factory.RemoteFactory.create", return_value=mock_remote
    ):
        with patch(
            "truss.cli.remote_cli.inquire_remote_name", return_value="inquired_remote"
        ) as mock_inquire:
            runner.invoke(truss_cli, ["loops", "deactivate", "--yes"])

    mock_inquire.assert_called_once()


def test_deactivate_propagates_error(mock_remote):
    mock_remote.deactivate_trainer_deployment.side_effect = RuntimeError(
        "deactivation failed"
    )

    result = _invoke_loops_deactivate(
        ["--remote", "test_remote", "--yes"], mock_remote
    )

    assert result.exit_code != 0
