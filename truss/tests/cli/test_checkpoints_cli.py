"""Tests for ``truss checkpoints`` CLI commands.

The ``checkpoints`` group is a Loops-flavored alias of the legacy
``truss train deploy_checkpoints`` / ``truss train checkpoints list``
surfaces. Both groups share the same backend routing logic in
``train_commands._run_deploy_checkpoints`` / ``_run_view_checkpoints``,
so these tests exercise that shared logic end-to-end via the new
``checkpoints`` entry point.
"""

import os
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from truss.cli.cli import truss_cli


@pytest.fixture
def mock_remote():
    remote = MagicMock()
    remote.api.list_loop_runs.return_value = [
        {"run_id": "trnr_xyz", "base_model": "Qwen/Qwen3-8B"}
    ]
    remote.api.list_loop_checkpoints.return_value = {
        "checkpoints": [
            {
                "id": "tcp_step100",
                "checkpoint_id": "step-100",
                "base_model": "Qwen/Qwen3-8B",
                "checkpoint_type": "lora",
                "size_bytes": 1024,
                "created_at": "2026-05-07T00:00:00Z",
            }
        ]
    }
    remote.api.list_loop_checkpoint_files.return_value = []
    return remote


def _invoke(args, mock_remote):
    env = os.environ.copy()
    env["COLUMNS"] = "200"
    runner = CliRunner(env=env)
    with patch(
        "truss.remote.remote_factory.RemoteFactory.create", return_value=mock_remote
    ):
        return runner.invoke(truss_cli, args)


def test_checkpoints_help_lists_subcommands():
    env = os.environ.copy()
    env["COLUMNS"] = "200"
    runner = CliRunner(env=env)
    result = runner.invoke(truss_cli, ["checkpoints", "--help"])
    assert result.exit_code == 0
    assert "deploy" in result.output
    assert "view" in result.output


def test_view_with_run_id_calls_list_loop_checkpoints(mock_remote):
    result = _invoke(
        [
            "checkpoints",
            "view",
            "--remote",
            "test_remote",
            "--run-id",
            "trnr_xyz",
            "--output-format",
            "json",
        ],
        mock_remote,
    )
    assert result.exit_code == 0, result.output
    mock_remote.api.list_loop_checkpoints.assert_called_once_with(
        run_id="trnr_xyz", base_model=None
    )


def test_view_with_model_name_calls_list_loop_checkpoints_with_base_model(mock_remote):
    result = _invoke(
        [
            "checkpoints",
            "view",
            "--remote",
            "test_remote",
            "--model-name",
            "Qwen/Qwen3-8B",
            "--output-format",
            "json",
        ],
        mock_remote,
    )
    assert result.exit_code == 0, result.output
    mock_remote.api.list_loop_checkpoints.assert_called_once_with(
        run_id=None, base_model="Qwen/Qwen3-8B"
    )


def test_view_rejects_run_id_combined_with_model_name(mock_remote):
    result = _invoke(
        [
            "checkpoints",
            "view",
            "--remote",
            "test_remote",
            "--run-id",
            "trnr_xyz",
            "--model-name",
            "Qwen/Qwen3-8B",
        ],
        mock_remote,
    )
    assert result.exit_code != 0
    assert "--run-id and --model-name cannot be combined" in result.output
    mock_remote.api.list_loop_checkpoints.assert_not_called()


def test_view_rejects_run_id_combined_with_project_id(mock_remote):
    result = _invoke(
        [
            "checkpoints",
            "view",
            "--remote",
            "test_remote",
            "--run-id",
            "trnr_xyz",
            "--project-id",
            "proj_abc",
        ],
        mock_remote,
    )
    assert result.exit_code != 0
    assert (
        "--run-id / --model-name cannot be combined with --project, "
        "--project-id, or --job-id" in result.output
    )
    mock_remote.api.list_loop_checkpoints.assert_not_called()


def test_view_with_run_id_and_checkpoint_name_drills_into_files(mock_remote):
    result = _invoke(
        [
            "checkpoints",
            "view",
            "--remote",
            "test_remote",
            "--run-id",
            "trnr_xyz",
            "--checkpoint-name",
            "step-100",
        ],
        mock_remote,
    )
    assert result.exit_code == 0, result.output
    # Drilldown fetches files via the Loops checkpoint files endpoint, keyed
    # by row PK (``id``), not user-facing checkpoint_id.
    mock_remote.api.list_loop_checkpoint_files.assert_called_once_with(
        checkpoint_id="tcp_step100"
    )


def test_view_with_unknown_checkpoint_name_prints_friendly_message(mock_remote):
    result = _invoke(
        [
            "checkpoints",
            "view",
            "--remote",
            "test_remote",
            "--run-id",
            "trnr_xyz",
            "--checkpoint-name",
            "step-NOPE",
        ],
        mock_remote,
    )
    assert result.exit_code == 0, result.output
    assert "No checkpoint named 'step-NOPE'" in result.output
    mock_remote.api.list_loop_checkpoint_files.assert_not_called()


def test_view_with_run_id_no_checkpoints_prints_empty_message(mock_remote):
    mock_remote.api.list_loop_checkpoints.return_value = {"checkpoints": []}
    result = _invoke(
        [
            "checkpoints",
            "view",
            "--remote",
            "test_remote",
            "--run-id",
            "trnr_xyz",
            "--output-format",
            "json",
        ],
        mock_remote,
    )
    assert result.exit_code == 0, result.output


def test_deploy_rejects_run_id_combined_with_project_id(mock_remote):
    result = _invoke(
        [
            "checkpoints",
            "deploy",
            "--remote",
            "test_remote",
            "--run-id",
            "trnr_xyz",
            "--project-id",
            "proj_abc",
        ],
        mock_remote,
    )
    assert result.exit_code != 0
    assert (
        "--run-id cannot be combined with --project, --project-id, or --job-id"
        in result.output
    )


def test_deploy_rejects_run_id_combined_with_job_id(mock_remote):
    result = _invoke(
        [
            "checkpoints",
            "deploy",
            "--remote",
            "test_remote",
            "--run-id",
            "trnr_xyz",
            "--job-id",
            "tj_abc",
        ],
        mock_remote,
    )
    assert result.exit_code != 0
    assert (
        "--run-id cannot be combined with --project, --project-id, or --job-id"
        in result.output
    )
