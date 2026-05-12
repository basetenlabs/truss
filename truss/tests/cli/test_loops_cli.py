"""Tests for truss loops CLI commands."""

import os
import re
from unittest.mock import Mock, patch

import pytest
from click.testing import CliRunner

from truss.cli.cli import truss_cli

_LOCALIZED_TIMESTAMP_RE = re.compile(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}")


@pytest.fixture
def mock_remote():
    remote = Mock()
    remote.create_loops_session.return_value = {"id": "session_abc123"}
    remote.create_loops_run.return_value = {
        "id": "abc123",
        "base_url": "https://trainer-xyz456.api.baseten.co/trainer",
        "sampler": {
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
    mock_remote.create_loops_session.assert_called_once_with(training_project_id=None)
    mock_remote.create_loops_run.assert_called_once_with(
        session_id="session_abc123", base_model="Qwen/Qwen3-8B"
    )
    assert "Qwen/Qwen3-8B" in result.output


def test_push_with_project_id(mock_remote):
    result = _invoke_loops_push(
        ["Qwen/Qwen3-8B", "--remote", "test_remote", "--project-id", "proj_abc"],
        mock_remote,
    )

    assert result.exit_code == 0, result.output
    mock_remote.create_loops_session.assert_called_once_with(
        training_project_id="proj_abc"
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
    mock_remote.create_loops_session.side_effect = RuntimeError(
        "session creation failed"
    )

    result = _invoke_loops_push(
        ["Qwen/Qwen3-8B", "--remote", "test_remote"], mock_remote
    )

    assert result.exit_code != 0


def test_push_propagates_loops_run_creation_error(mock_remote):
    mock_remote.create_loops_run.side_effect = RuntimeError(
        "active Loops deployment already exists"
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


def _invoke_loops_deactivate(args, mock_remote, input=None):
    runner = CliRunner()
    with patch(
        "truss.remote.remote_factory.RemoteFactory.create", return_value=mock_remote
    ):
        return runner.invoke(truss_cli, ["loops", "deactivate"] + args, input=input)


def test_deactivate_basic(mock_remote):
    result = _invoke_loops_deactivate(
        ["Qwen/Qwen3-8B", "--remote", "test_remote", "--yes"], mock_remote
    )

    assert result.exit_code == 0, result.output
    mock_remote.deactivate_loops_deployment.assert_called_once_with("Qwen/Qwen3-8B")
    assert "deactivated" in result.output


def test_deactivate_confirms_before_proceeding(mock_remote):
    result = _invoke_loops_deactivate(
        ["Qwen/Qwen3-8B", "--remote", "test_remote"], mock_remote, input="y\n"
    )

    assert result.exit_code == 0, result.output
    mock_remote.deactivate_loops_deployment.assert_called_once_with("Qwen/Qwen3-8B")


def test_deactivate_aborts_on_no_confirmation(mock_remote):
    result = _invoke_loops_deactivate(
        ["Qwen/Qwen3-8B", "--remote", "test_remote"], mock_remote, input="n\n"
    )

    assert result.exit_code != 0
    mock_remote.deactivate_loops_deployment.assert_not_called()


def test_deactivate_uses_inquire_when_remote_not_provided(mock_remote):
    runner = CliRunner()
    with patch(
        "truss.remote.remote_factory.RemoteFactory.create", return_value=mock_remote
    ):
        with patch(
            "truss.cli.remote_cli.inquire_remote_name", return_value="inquired_remote"
        ) as mock_inquire:
            runner.invoke(truss_cli, ["loops", "deactivate", "Qwen/Qwen3-8B", "--yes"])

    mock_inquire.assert_called_once()


def test_deactivate_propagates_error(mock_remote):
    mock_remote.deactivate_loops_deployment.side_effect = RuntimeError(
        "deactivation failed"
    )

    result = _invoke_loops_deactivate(
        ["Qwen/Qwen3-8B", "--remote", "test_remote", "--yes"], mock_remote
    )

    assert result.exit_code != 0


def test_deactivate_requires_base_model(mock_remote):
    result = _invoke_loops_deactivate(["--remote", "test_remote", "--yes"], mock_remote)

    assert result.exit_code != 0
    mock_remote.deactivate_loops_deployment.assert_not_called()


def _invoke(args, mock_remote):
    env = os.environ.copy()
    env["COLUMNS"] = "200"
    runner = CliRunner(env=env)
    with patch(
        "truss.remote.remote_factory.RemoteFactory.create", return_value=mock_remote
    ):
        return runner.invoke(truss_cli, args)


def test_view_lists_active_deployments(mock_remote):
    mock_remote.api.list_loops_deployments.return_value = [
        {
            "id": "dep_abc",
            "base_model": "Qwen/Qwen3-8B",
            "base_url": "https://trainer-abc.api.baseten.co/trainer",
            "sampler": {
                "id": "sampler_def",
                "base_url": "https://model-def.api.baseten.co/deployment/v1/sync",
            },
        }
    ]
    result = _invoke(["loops", "view", "--remote", "test_remote"], mock_remote)
    assert result.exit_code == 0, result.output
    assert "dep_abc" in result.output
    assert "Qwen/Qwen3-8B" in result.output
    assert "model-def.api.baseten.co" in result.output


def test_view_with_no_deployments_prints_friendly_message(mock_remote):
    mock_remote.api.list_loops_deployments.return_value = []
    result = _invoke(["loops", "view", "--remote", "test_remote"], mock_remote)
    assert result.exit_code == 0, result.output
    assert "No active Loops deployments" in result.output


def test_runs_view_no_filters_calls_search_with_none(mock_remote):
    mock_remote.api.list_loops_runs.return_value = [
        {"id": "trnr_xyz", "session_id": "sess_abc", "base_model": "Qwen/Qwen3-8B"}
    ]
    result = _invoke(["loops", "runs", "view", "--remote", "test_remote"], mock_remote)
    assert result.exit_code == 0, result.output
    mock_remote.api.list_loops_runs.assert_called_once_with(
        run_id=None, base_model=None
    )
    assert "trnr_xyz" in result.output


def test_runs_view_with_run_id_filter(mock_remote):
    mock_remote.api.list_loops_runs.return_value = [
        {"id": "trnr_xyz", "session_id": "sess_abc", "base_model": "Qwen/Qwen3-8B"}
    ]
    result = _invoke(
        ["loops", "runs", "view", "--remote", "test_remote", "--run-id", "trnr_xyz"],
        mock_remote,
    )
    assert result.exit_code == 0, result.output
    mock_remote.api.list_loops_runs.assert_called_once_with(
        run_id="trnr_xyz", base_model=None
    )


def test_runs_view_with_base_model_filter(mock_remote):
    mock_remote.api.list_loops_runs.return_value = []
    result = _invoke(
        [
            "loops",
            "runs",
            "view",
            "--remote",
            "test_remote",
            "--base-model",
            "Qwen/Qwen3-8B",
        ],
        mock_remote,
    )
    assert result.exit_code == 0, result.output
    mock_remote.api.list_loops_runs.assert_called_once_with(
        run_id=None, base_model="Qwen/Qwen3-8B"
    )
    assert "No Loops runs found" in result.output


def test_samplers_view_lists_samplers(mock_remote):
    mock_remote.api.list_loops_samplers.return_value = [
        {
            "id": "sampler_def",
            "base_url": "https://model-def.api.baseten.co/deployment/v1/sync",
        }
    ]
    result = _invoke(
        ["loops", "samplers", "view", "--remote", "test_remote"], mock_remote
    )
    assert result.exit_code == 0, result.output
    mock_remote.api.list_loops_samplers.assert_called_once_with()
    assert "sampler_def" in result.output


def test_samplers_view_no_samplers_prints_friendly_message(mock_remote):
    mock_remote.api.list_loops_samplers.return_value = []
    result = _invoke(
        ["loops", "samplers", "view", "--remote", "test_remote"], mock_remote
    )
    assert result.exit_code == 0, result.output
    assert "No Loops samplers found" in result.output


def test_runs_view_renders_base_url_and_localized_created_at(mock_remote):
    mock_remote.api.list_loops_runs.return_value = [
        {
            "id": "trnr_xyz",
            "session_id": "sess_abc",
            "base_model": "Qwen/Qwen3-8B",
            "base_url": "https://trainer-xyz.api.baseten.co/trainer",
            "created_at": "2026-05-07T12:34:56Z",
        }
    ]
    result = _invoke(["loops", "runs", "view", "--remote", "test_remote"], mock_remote)
    assert result.exit_code == 0, result.output
    assert "trainer-xyz.api.baseten.co" in result.output
    # The raw ISO suffix should be replaced by the localized format.
    assert "2026-05-07T12:34:56Z" not in result.output
    assert _LOCALIZED_TIMESTAMP_RE.search(result.output) is not None


def test_runs_view_default_puts_newest_at_bottom(mock_remote):
    mock_remote.api.list_loops_runs.return_value = [
        {
            "id": "trnr_old",
            "session_id": "sess_a",
            "base_model": "Qwen/Qwen3-8B",
            "created_at": "2026-05-01T00:00:00Z",
        },
        {
            "id": "trnr_new",
            "session_id": "sess_b",
            "base_model": "Qwen/Qwen3-8B",
            "created_at": "2026-05-07T00:00:00Z",
        },
    ]
    result = _invoke(["loops", "runs", "view", "--remote", "test_remote"], mock_remote)
    assert result.exit_code == 0, result.output
    assert result.output.index("trnr_old") < result.output.index("trnr_new")


def test_runs_view_reverse_puts_newest_first(mock_remote):
    mock_remote.api.list_loops_runs.return_value = [
        {
            "id": "trnr_old",
            "session_id": "sess_a",
            "base_model": "Qwen/Qwen3-8B",
            "created_at": "2026-05-01T00:00:00Z",
        },
        {
            "id": "trnr_new",
            "session_id": "sess_b",
            "base_model": "Qwen/Qwen3-8B",
            "created_at": "2026-05-07T00:00:00Z",
        },
    ]
    result = _invoke(
        ["loops", "runs", "view", "--remote", "test_remote", "--reverse"], mock_remote
    )
    assert result.exit_code == 0, result.output
    assert result.output.index("trnr_new") < result.output.index("trnr_old")


def test_samplers_view_default_puts_newest_at_bottom(mock_remote):
    mock_remote.api.list_loops_samplers.return_value = [
        {
            "id": "sampler_old",
            "base_model": "Qwen/Qwen3-8B",
            "base_url": "https://model-old.api.baseten.co/deployment/v1/sync",
            "created_at": "2026-05-01T00:00:00Z",
        },
        {
            "id": "sampler_new",
            "base_model": "Qwen/Qwen3-8B",
            "base_url": "https://model-new.api.baseten.co/deployment/v1/sync",
            "created_at": "2026-05-07T00:00:00Z",
        },
    ]
    result = _invoke(
        ["loops", "samplers", "view", "--remote", "test_remote"], mock_remote
    )
    assert result.exit_code == 0, result.output
    assert result.output.index("sampler_old") < result.output.index("sampler_new")
    # Verify localized timestamp formatting (ISO suffix replaced).
    assert "2026-05-01T00:00:00Z" not in result.output
    assert _LOCALIZED_TIMESTAMP_RE.search(result.output) is not None


def test_samplers_view_reverse_puts_newest_first(mock_remote):
    mock_remote.api.list_loops_samplers.return_value = [
        {
            "id": "sampler_old",
            "base_model": "Qwen/Qwen3-8B",
            "base_url": "https://model-old.api.baseten.co/deployment/v1/sync",
            "created_at": "2026-05-01T00:00:00Z",
        },
        {
            "id": "sampler_new",
            "base_model": "Qwen/Qwen3-8B",
            "base_url": "https://model-new.api.baseten.co/deployment/v1/sync",
            "created_at": "2026-05-07T00:00:00Z",
        },
    ]
    result = _invoke(
        ["loops", "samplers", "view", "--remote", "test_remote", "--reverse"],
        mock_remote,
    )
    assert result.exit_code == 0, result.output
    assert result.output.index("sampler_new") < result.output.index("sampler_old")


def test_checkpoints_view_requires_run_id_or_base_model(mock_remote):
    result = _invoke(
        ["loops", "checkpoints", "view", "--remote", "test_remote"], mock_remote
    )
    assert result.exit_code != 0
    mock_remote.api.list_loops_checkpoints.assert_not_called()
    mock_remote.api.list_loops_runs.assert_not_called()


def test_checkpoints_view_rejects_both_run_id_and_base_model(mock_remote):
    result = _invoke(
        [
            "loops",
            "checkpoints",
            "view",
            "--remote",
            "test_remote",
            "--run-id",
            "trnr_xyz",
            "--base-model",
            "Qwen/Qwen3-8B",
        ],
        mock_remote,
    )
    assert result.exit_code != 0
    mock_remote.api.list_loops_checkpoints.assert_not_called()
    mock_remote.api.list_loops_runs.assert_not_called()


def test_checkpoints_view_with_run_id_calls_list_loops_checkpoints(mock_remote):
    mock_remote.api.list_loops_checkpoints.return_value = {
        "checkpoints": [
            {
                "id": "tcp_step100",
                "checkpoint_id": "step-100",
                "checkpoint_type": "lora",
                "base_model": "Qwen/Qwen3-8B",
                "size_bytes": 1234,
                "created_at": "2026-05-07T12:34:56Z",
            }
        ]
    }
    result = _invoke(
        [
            "loops",
            "checkpoints",
            "view",
            "--remote",
            "test_remote",
            "--run-id",
            "trnr_xyz",
        ],
        mock_remote,
    )
    assert result.exit_code == 0, result.output
    mock_remote.api.list_loops_checkpoints.assert_called_once_with(run_id="trnr_xyz")
    assert "step-100" in result.output  # checkpoint name
    assert "tcp_step100" in result.output  # Loops checkpoint PK
    assert "trnr_xyz" in result.output  # table title shows the run id


def test_checkpoints_view_with_base_model_picks_most_recent_run(mock_remote):
    mock_remote.api.list_loops_runs.return_value = [
        {"id": "trnr_old", "created_at": "2026-05-01T00:00:00Z"},
        {"id": "trnr_new", "created_at": "2026-05-07T00:00:00Z"},
    ]
    mock_remote.api.list_loops_checkpoints.return_value = {"checkpoints": []}
    result = _invoke(
        [
            "loops",
            "checkpoints",
            "view",
            "--remote",
            "test_remote",
            "--base-model",
            "Qwen/Qwen3-8B",
        ],
        mock_remote,
    )
    assert result.exit_code == 0, result.output
    mock_remote.api.list_loops_runs.assert_called_once_with(base_model="Qwen/Qwen3-8B")
    mock_remote.api.list_loops_checkpoints.assert_called_once_with(run_id="trnr_new")


def test_checkpoints_view_base_model_no_runs(mock_remote):
    mock_remote.api.list_loops_runs.return_value = []
    result = _invoke(
        [
            "loops",
            "checkpoints",
            "view",
            "--remote",
            "test_remote",
            "--base-model",
            "Qwen/Qwen3-8B",
        ],
        mock_remote,
    )
    assert result.exit_code != 0
    mock_remote.api.list_loops_checkpoints.assert_not_called()


def test_checkpoints_view_json_format_emits_run_id_key(mock_remote):
    mock_remote.api.list_loops_checkpoints.return_value = {
        "checkpoints": [
            {
                "id": "tcp_step100",
                "checkpoint_id": "step-100",
                "checkpoint_type": "lora",
                "base_model": "Qwen/Qwen3-8B",
                "size_bytes": 1234,
                "created_at": "2026-05-07T12:34:56Z",
            }
        ]
    }
    result = _invoke(
        [
            "loops",
            "checkpoints",
            "view",
            "--remote",
            "test_remote",
            "--run-id",
            "trnr_xyz",
            "-o",
            "json",
        ],
        mock_remote,
    )
    assert result.exit_code == 0, result.output
    assert '"run_id": "trnr_xyz"' in result.output
    assert '"job_id"' not in result.output
    assert '"id": "tcp_step100"' in result.output


def test_checkpoints_deploy_requires_run_id_or_config(mock_remote):
    with patch(
        "truss.cli.loops_commands.train_cli.create_model_version_from_inference_template"
    ) as mock_create:
        result = _invoke(
            ["loops", "checkpoints", "deploy", "--remote", "test_remote"], mock_remote
        )
    assert result.exit_code != 0
    mock_create.assert_not_called()


def test_checkpoints_deploy_with_run_id_invokes_shared_path(mock_remote):
    with (
        patch(
            "truss.cli.loops_commands.train_cli.create_model_version_from_inference_template"
        ) as mock_create,
        patch("truss.cli.loops_commands.train_cli.write_truss_config"),
        patch(
            "truss.cli.loops_commands.train_cli.print_deploy_checkpoints_success_message"
        ),
    ):
        mock_create.return_value = Mock(deploy_config=Mock(), truss_config=None)
        result = _invoke(
            [
                "loops",
                "checkpoints",
                "deploy",
                "--remote",
                "test_remote",
                "--run-id",
                "trnr_xyz",
                "--dry-run",
            ],
            mock_remote,
        )
    assert result.exit_code == 0, result.output
    args = mock_create.call_args
    deploy_args = args[0][1]
    assert deploy_args.run_id == "trnr_xyz"
    assert deploy_args.project_id is None
    assert deploy_args.job_id is None
    assert deploy_args.is_loops_command is True
    assert deploy_args.dry_run is True


def test_checkpoints_deploy_with_checkpoint_ids_parses_and_forwards(mock_remote):
    with (
        patch(
            "truss.cli.loops_commands.train_cli.create_model_version_from_inference_template"
        ) as mock_create,
        patch("truss.cli.loops_commands.train_cli.write_truss_config"),
        patch(
            "truss.cli.loops_commands.train_cli.print_deploy_checkpoints_success_message"
        ),
    ):
        mock_create.return_value = Mock(deploy_config=Mock(), truss_config=None)
        result = _invoke(
            [
                "loops",
                "checkpoints",
                "deploy",
                "--remote",
                "test_remote",
                "--checkpoint-ids",
                "tcp_step100, tcp_step200 ,tcp_step300",
                "--dry-run",
            ],
            mock_remote,
        )
    assert result.exit_code == 0, result.output
    deploy_args = mock_create.call_args[0][1]
    assert deploy_args.checkpoint_ids == ["tcp_step100", "tcp_step200", "tcp_step300"]
    assert deploy_args.run_id is None
    assert deploy_args.deploy_config_path is None


def test_checkpoints_deploy_rejects_checkpoint_ids_with_config(mock_remote, tmp_path):
    config_path = tmp_path / "deploy.py"
    config_path.write_text("")
    with patch(
        "truss.cli.loops_commands.train_cli.create_model_version_from_inference_template"
    ) as mock_create:
        result = _invoke(
            [
                "loops",
                "checkpoints",
                "deploy",
                "--remote",
                "test_remote",
                "--checkpoint-ids",
                "tcp_step100",
                "--config",
                str(config_path),
            ],
            mock_remote,
        )
    assert result.exit_code != 0
    mock_create.assert_not_called()
