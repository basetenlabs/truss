"""Tests for truss loops CLI commands."""

import json
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
        return runner.invoke(truss_cli, ["loops", "push"] + args)


def test_push_basic(mock_remote):
    result = _invoke_loops_push(
        ["Qwen/Qwen3-8B", "--remote", "test_remote"], mock_remote
    )

    assert result.exit_code == 0, result.output
    mock_remote.create_loops_session.assert_called_once_with(training_project_id=None)
    mock_remote.create_loops_run.assert_called_once_with(
        session_id="session_abc123", base_model="Qwen/Qwen3-8B", replicas=None
    )
    assert "Qwen/Qwen3-8B" in result.output


def test_push_with_replicas(mock_remote):
    result = _invoke_loops_push(
        ["Qwen/Qwen3-8B", "--remote", "test_remote", "--replicas", "4"], mock_remote
    )

    assert result.exit_code == 0, result.output
    mock_remote.create_loops_run.assert_called_once_with(
        session_id="session_abc123", base_model="Qwen/Qwen3-8B", replicas=4
    )


def test_push_rejects_non_positive_replicas(mock_remote):
    result = _invoke_loops_push(
        ["Qwen/Qwen3-8B", "--remote", "test_remote", "--replicas", "0"], mock_remote
    )

    assert result.exit_code != 0
    mock_remote.create_loops_run.assert_not_called()


def test_push_with_project_id(mock_remote):
    result = _invoke_loops_push(
        ["Qwen/Qwen3-8B", "--remote", "test_remote", "--project-id", "proj_abc"],
        mock_remote,
    )

    assert result.exit_code == 0, result.output
    mock_remote.create_loops_session.assert_called_once_with(
        training_project_id="proj_abc"
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
        ["dep_abc", "--remote", "test_remote", "--yes"], mock_remote
    )

    assert result.exit_code == 0, result.output
    mock_remote.api.deactivate_loops_deployment.assert_called_once_with("dep_abc")
    assert "deactivated" in result.output


def test_deactivate_confirms_before_proceeding(mock_remote):
    result = _invoke_loops_deactivate(
        ["dep_abc", "--remote", "test_remote"], mock_remote, input="y\n"
    )

    assert result.exit_code == 0, result.output
    mock_remote.api.deactivate_loops_deployment.assert_called_once_with("dep_abc")


def test_deactivate_aborts_on_no_confirmation(mock_remote):
    result = _invoke_loops_deactivate(
        ["dep_abc", "--remote", "test_remote"], mock_remote, input="n\n"
    )

    assert result.exit_code != 0
    mock_remote.api.deactivate_loops_deployment.assert_not_called()


def test_deactivate_uses_inquire_when_remote_not_provided(mock_remote):
    runner = CliRunner()
    with patch(
        "truss.remote.remote_factory.RemoteFactory.create", return_value=mock_remote
    ):
        with patch(
            "truss.cli.remote_cli.inquire_remote_name", return_value="inquired_remote"
        ) as mock_inquire:
            runner.invoke(truss_cli, ["loops", "deactivate", "dep_abc", "--yes"])

    mock_inquire.assert_called_once()


def test_deactivate_propagates_error(mock_remote):
    mock_remote.api.deactivate_loops_deployment.side_effect = RuntimeError(
        "deactivation failed"
    )

    result = _invoke_loops_deactivate(
        ["dep_abc", "--remote", "test_remote", "--yes"], mock_remote
    )

    assert result.exit_code != 0


def test_deactivate_requires_deployment_id(mock_remote):
    result = _invoke_loops_deactivate(["--remote", "test_remote", "--yes"], mock_remote)

    assert result.exit_code != 0
    mock_remote.api.deactivate_loops_deployment.assert_not_called()


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
            "status": {"name": "RUNNING"},
            "sampler": {
                "id": "sampler_def",
                "deployment_id": "ov_def123",
                "base_url": "https://model-def.api.baseten.co/deployment/v1/sync",
                "status": {"name": "ACTIVE"},
            },
        }
    ]
    result = _invoke(["loops", "view", "--remote", "test_remote"], mock_remote)
    assert result.exit_code == 0, result.output
    assert "dep_abc" in result.output
    assert "Qwen/Qwen3-8B" in result.output
    assert "RUNNING" in result.output
    assert "ACTIVE" in result.output
    assert "ov_def123" in result.output
    assert "model-def.api.baseten.co" in result.output


def test_view_with_no_deployments_prints_friendly_message(mock_remote):
    mock_remote.api.list_loops_deployments.return_value = []
    result = _invoke(["loops", "view", "--remote", "test_remote"], mock_remote)
    assert result.exit_code == 0, result.output
    assert "No Loops deployments" in result.output
    # Truly empty: don't suggest --all since there's nothing to reveal.
    assert "--all" not in result.output


def _deployment(deployment_id: str, status_name: str) -> dict:
    return {
        "id": deployment_id,
        "base_model": "Qwen/Qwen3-8B",
        "base_url": f"https://trainer-{deployment_id}.api.baseten.co/trainer",
        "status": {"name": status_name},
        "sampler": {
            "id": f"sampler_{deployment_id}",
            "deployment_id": f"ov_{deployment_id}",
            "base_url": f"https://model-{deployment_id}.api.baseten.co/deployment/v1/sync",
            "status": {"name": "ACTIVE"},
        },
    }


def test_view_filters_stopped_and_failed_by_default(mock_remote):
    mock_remote.api.list_loops_deployments.return_value = [
        _deployment("dep_running", "RUNNING"),
        _deployment("dep_stopped", "STOPPED"),
        _deployment("dep_failed", "FAILED"),
        _deployment("dep_deploying", "DEPLOYING"),
    ]
    result = _invoke(["loops", "view", "--remote", "test_remote"], mock_remote)
    assert result.exit_code == 0, result.output
    assert "dep_running" in result.output
    assert "dep_deploying" in result.output
    assert "dep_stopped" not in result.output
    assert "dep_failed" not in result.output


def test_view_all_flag_includes_terminal_states(mock_remote):
    mock_remote.api.list_loops_deployments.return_value = [
        _deployment("dep_running", "RUNNING"),
        _deployment("dep_stopped", "STOPPED"),
        _deployment("dep_failed", "FAILED"),
    ]
    result = _invoke(["loops", "view", "--all", "--remote", "test_remote"], mock_remote)
    assert result.exit_code == 0, result.output
    assert "dep_running" in result.output
    assert "dep_stopped" in result.output
    assert "dep_failed" in result.output


def test_view_empty_after_filter_hints_at_all_flag(mock_remote):
    mock_remote.api.list_loops_deployments.return_value = [
        _deployment("dep_stopped", "STOPPED"),
        _deployment("dep_failed", "FAILED"),
    ]
    result = _invoke(["loops", "view", "--remote", "test_remote"], mock_remote)
    assert result.exit_code == 0, result.output
    assert "No active Loops deployments" in result.output
    assert "--all" in result.output


def test_view_all_flag_with_no_deployments(mock_remote):
    mock_remote.api.list_loops_deployments.return_value = []
    result = _invoke(["loops", "view", "--all", "--remote", "test_remote"], mock_remote)
    assert result.exit_code == 0, result.output
    assert "No Loops deployments" in result.output
    assert "--all" not in result.output


def _parse_jsonl(output: str) -> list[dict]:
    return [json.loads(line) for line in output.splitlines() if line.strip()]


def test_view_json_output_emits_one_object_per_deployment(mock_remote):
    mock_remote.api.list_loops_deployments.return_value = [
        {
            "id": "dep_abc",
            "base_model": "Qwen/Qwen3-8B",
            "base_url": "https://trainer-abc.api.baseten.co/trainer",
            "status": {"name": "RUNNING"},
            "sampler": {
                "id": "sampler_def",
                "deployment_id": "ov_def123",
                "base_url": "https://model-def.api.baseten.co/deployment/v1/sync",
                "status": {"name": "ACTIVE"},
            },
        }
    ]
    result = _invoke(
        ["loops", "view", "--remote", "test_remote", "-o", "json"], mock_remote
    )
    assert result.exit_code == 0, result.output
    records = _parse_jsonl(result.output)
    assert records == [
        {
            "id": "dep_abc",
            "base_model": "Qwen/Qwen3-8B",
            "base_url": "https://trainer-abc.api.baseten.co/trainer",
            "status": "RUNNING",
            "sampler": {
                "deployment_id": "ov_def123",
                "base_url": "https://model-def.api.baseten.co/deployment/v1/sync",
                "status": "ACTIVE",
            },
        }
    ]


def test_view_json_output_with_no_deployments_emits_nothing(mock_remote):
    # JSONL stream of zero records: no stdout content, and crucially no
    # friendly "No Loops deployments." message that would corrupt the stream.
    mock_remote.api.list_loops_deployments.return_value = []
    result = _invoke(
        ["loops", "view", "--remote", "test_remote", "-o", "json"], mock_remote
    )
    assert result.exit_code == 0, result.output
    assert result.output.strip() == ""


def test_view_json_output_filters_terminal_states_by_default(mock_remote):
    mock_remote.api.list_loops_deployments.return_value = [
        _deployment("dep_running", "RUNNING"),
        _deployment("dep_stopped", "STOPPED"),
        _deployment("dep_failed", "FAILED"),
    ]
    result = _invoke(
        ["loops", "view", "--remote", "test_remote", "-o", "json"], mock_remote
    )
    assert result.exit_code == 0, result.output
    records = _parse_jsonl(result.output)
    assert [r["id"] for r in records] == ["dep_running"]


def test_view_json_output_filter_to_empty_emits_nothing(mock_remote):
    # Raw list non-empty but the default terminal-state filter empties it;
    # JSON consumers should get an empty stream — no "pass --all" hint that
    # the CLI table prints.
    mock_remote.api.list_loops_deployments.return_value = [
        _deployment("dep_stopped", "STOPPED"),
        _deployment("dep_failed", "FAILED"),
    ]
    result = _invoke(
        ["loops", "view", "--remote", "test_remote", "-o", "json"], mock_remote
    )
    assert result.exit_code == 0, result.output
    assert result.output.strip() == ""
    assert "--all" not in result.output


def test_view_json_output_all_flag_includes_terminal_states(mock_remote):
    mock_remote.api.list_loops_deployments.return_value = [
        _deployment("dep_running", "RUNNING"),
        _deployment("dep_stopped", "STOPPED"),
    ]
    result = _invoke(
        ["loops", "view", "--all", "--remote", "test_remote", "-o", "json"], mock_remote
    )
    assert result.exit_code == 0, result.output
    records = _parse_jsonl(result.output)
    assert sorted(r["id"] for r in records) == ["dep_running", "dep_stopped"]


def test_view_renders_deployment_with_null_sampler(mock_remote):
    # Backend surfaces orphaned deployments with ``sampler: null`` rather
    # than dropping them. The table must render the row with placeholders
    # instead of KeyError'ing on sampler["..."].
    mock_remote.api.list_loops_deployments.return_value = [
        {
            "id": "dep_orphan",
            "base_model": "Qwen/Qwen3-8B",
            "base_url": "https://trainer-orphan.api.baseten.co/trainer",
            "status": {"name": "RUNNING"},
            "sampler": None,
        }
    ]
    result = _invoke(["loops", "view", "--remote", "test_remote"], mock_remote)
    assert result.exit_code == 0, result.output
    assert "dep_orphan" in result.output


def test_view_json_output_renders_null_sampler(mock_remote):
    mock_remote.api.list_loops_deployments.return_value = [
        {
            "id": "dep_orphan",
            "base_model": "Qwen/Qwen3-8B",
            "base_url": "https://trainer-orphan.api.baseten.co/trainer",
            "status": {"name": "RUNNING"},
            "sampler": None,
        }
    ]
    result = _invoke(
        ["loops", "view", "--remote", "test_remote", "-o", "json"], mock_remote
    )
    assert result.exit_code == 0, result.output
    records = _parse_jsonl(result.output)
    assert records == [
        {
            "id": "dep_orphan",
            "base_model": "Qwen/Qwen3-8B",
            "base_url": "https://trainer-orphan.api.baseten.co/trainer",
            "status": "RUNNING",
            "sampler": None,
        }
    ]


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
                "id": "vL3pQrS8",
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
    assert "vL3pQrS8" in result.output  # Loops checkpoint PK
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
                "id": "vL3pQrS8",
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
    assert '"id": "vL3pQrS8"' in result.output


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
                "vL3pQrS8, wK4tUvW9 ,xM5yZaB0",
                "--dry-run",
            ],
            mock_remote,
        )
    assert result.exit_code == 0, result.output
    deploy_args = mock_create.call_args[0][1]
    assert deploy_args.checkpoint_ids == ["vL3pQrS8", "wK4tUvW9", "xM5yZaB0"]
    assert deploy_args.run_id is None
    assert deploy_args.deploy_config_path is None


def test_checkpoints_deploy_rejects_checkpoint_ids_with_run_id(mock_remote):
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
                "--run-id",
                "trnr_xyz",
                "--checkpoint-ids",
                "vL3pQrS8",
            ],
            mock_remote,
        )
    assert result.exit_code != 0
    mock_create.assert_not_called()


def test_checkpoints_deploy_rejects_whitespace_only_checkpoint_ids(mock_remote):
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
                " , ,",
            ],
            mock_remote,
        )
    assert result.exit_code != 0
    mock_create.assert_not_called()


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
                "vL3pQrS8",
                "--config",
                str(config_path),
            ],
            mock_remote,
        )
    assert result.exit_code != 0
    mock_create.assert_not_called()


# ---------------------------------------------------------------------------
# `truss loops runs metrics` — request volume / concurrency per run window.
# ---------------------------------------------------------------------------


def _series(values, start_ts=1700000000):
    return [{"timestamp": start_ts + i * 15, "value": v} for i, v in enumerate(values)]


def _setup_run_metrics_mocks(
    mock_remote,
    *,
    trainer_volume,
    trainer_concurrent,
    sampler_volume=(),
    sampler_concurrent=(),
    sampler_deployment_id="ov_sampler",
    run_base_model="Qwen/Qwen3-8B",
    deployments_base_models=None,
):
    """Wire up the four API methods the metrics command calls.

    `deployments_base_models` is a list of (deployment_id, base_model, status_name);
    if omitted, defaults to one active deployment matching the run's base model.
    """
    mock_remote.api.get_loops_run.return_value = {
        "id": "trnr_xyz",
        "base_model": run_base_model,
        "created_at": "2026-05-07T12:34:56Z",
        "sampler": (
            {"deployment_id": sampler_deployment_id} if sampler_deployment_id else None
        ),
    }
    if deployments_base_models is None:
        deployments_base_models = [("dep_trainer", run_base_model, "RUNNING")]
    mock_remote.api.list_loops_deployments.return_value = [
        {"id": dep_id, "base_model": bm, "status": {"name": status}}
        for dep_id, bm, status in deployments_base_models
    ]
    mock_remote.api.get_loops_deployment_metrics.return_value = {
        "metrics": {
            "inference_volume": _series(trainer_volume),
            "concurrent_requests": _series(trainer_concurrent),
        }
    }
    mock_remote.api.get_model_deployment_range_metrics.return_value = {
        "inference_volume": _series(list(sampler_volume)),
        "model_concurrent_requests": _series(list(sampler_concurrent)),
    }


def test_runs_metrics_snapshot_renders_table_with_trainer_and_sampler(mock_remote):
    _setup_run_metrics_mocks(
        mock_remote,
        trainer_volume=[1.0, 2.5, 3.0],
        trainer_concurrent=[1.0, 2.0, 1.5],
        sampler_volume=[0.2, 0.4, 0.8],
        sampler_concurrent=[1.0, 1.0, 2.0],
    )
    result = _invoke(
        ["loops", "runs", "metrics", "--remote", "test_remote", "--run-id", "trnr_xyz"],
        mock_remote,
    )
    assert result.exit_code == 0, result.output
    assert "Trainer" in result.output
    assert "Sampler" in result.output
    assert "dep_trainer" in result.output
    assert "ov_sampler" in result.output
    # Latest values surface in the right-justified value columns.
    assert "3.00" in result.output  # trainer request volume latest
    assert "0.80" in result.output  # sampler request volume latest


def test_runs_metrics_snapshot_json_emits_single_document(mock_remote):
    _setup_run_metrics_mocks(
        mock_remote,
        trainer_volume=[1.0, 2.0],
        trainer_concurrent=[3.0, 4.0],
        sampler_volume=[0.5],
        sampler_concurrent=[1.5],
    )
    result = _invoke(
        [
            "loops",
            "runs",
            "metrics",
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
    records = _parse_jsonl(result.output)
    assert len(records) == 1
    doc = records[0]
    assert doc["run_id"] == "trnr_xyz"
    assert doc["trainer_deployment_id"] == "dep_trainer"
    assert doc["sampler_deployment_id"] == "ov_sampler"
    assert doc["trainer"]["request_volume"][-1]["value"] == 2.0
    assert doc["trainer"]["concurrent_requests"][-1]["value"] == 4.0
    assert doc["sampler"]["request_volume"][-1]["value"] == 0.5
    assert doc["sampler"]["concurrent_requests"][-1]["value"] == 1.5
    assert "start" in doc["window"] and "end" in doc["window"]


def test_runs_metrics_json_output_file_writes_to_path_not_stdout(mock_remote, tmp_path):
    _setup_run_metrics_mocks(
        mock_remote, trainer_volume=[1.0], trainer_concurrent=[2.0]
    )
    out = tmp_path / "metrics.json"
    result = _invoke(
        [
            "loops",
            "runs",
            "metrics",
            "--remote",
            "test_remote",
            "--run-id",
            "trnr_xyz",
            "-o",
            "json",
            "--output-file",
            str(out),
        ],
        mock_remote,
    )
    assert result.exit_code == 0, result.output
    # stdout stays empty so it's safe to redirect; payload lands in the file.
    assert result.output.strip() == ""
    written = out.read_text().splitlines()
    assert len(written) == 1
    doc = json.loads(written[0])
    assert doc["run_id"] == "trnr_xyz"


def test_runs_metrics_since_and_start_are_mutually_exclusive(mock_remote):
    result = _invoke(
        [
            "loops",
            "runs",
            "metrics",
            "--remote",
            "test_remote",
            "--run-id",
            "trnr_xyz",
            "--since",
            "1h",
            "--start",
            "2026-06-10T00:00:00Z",
        ],
        mock_remote,
    )
    assert result.exit_code != 0
    assert "--since and --start are mutually exclusive" in result.output


def test_runs_metrics_output_file_requires_json(mock_remote):
    result = _invoke(
        [
            "loops",
            "runs",
            "metrics",
            "--remote",
            "test_remote",
            "--run-id",
            "trnr_xyz",
            "--output-file",
            "/tmp/x.json",
        ],
        mock_remote,
    )
    assert result.exit_code != 0
    assert "--output-file requires -o json" in result.output


def test_runs_metrics_invalid_duration_is_rejected(mock_remote):
    _setup_run_metrics_mocks(
        mock_remote, trainer_volume=[1.0], trainer_concurrent=[2.0]
    )
    result = _invoke(
        [
            "loops",
            "runs",
            "metrics",
            "--remote",
            "test_remote",
            "--run-id",
            "trnr_xyz",
            "--since",
            "5x",
        ],
        mock_remote,
    )
    assert result.exit_code != 0
    assert "Invalid duration" in result.output


def test_runs_metrics_no_active_deployment_for_base_model_errors(mock_remote):
    _setup_run_metrics_mocks(
        mock_remote,
        trainer_volume=[],
        trainer_concurrent=[],
        deployments_base_models=[("dep_other", "OtherModel", "RUNNING")],
    )
    result = _invoke(
        ["loops", "runs", "metrics", "--remote", "test_remote", "--run-id", "trnr_xyz"],
        mock_remote,
    )
    assert result.exit_code != 0
    assert "No active Loops deployment found for base model" in result.output


def test_runs_metrics_ignores_inactive_deployments_when_resolving(mock_remote):
    # A STOPPED deployment with the same base_model must not be picked as the
    # trainer — only active ones qualify.
    _setup_run_metrics_mocks(
        mock_remote,
        trainer_volume=[1.0, 2.0],
        trainer_concurrent=[1.0, 1.0],
        deployments_base_models=[
            ("dep_old", "Qwen/Qwen3-8B", "STOPPED"),
            ("dep_trainer", "Qwen/Qwen3-8B", "RUNNING"),
        ],
    )
    result = _invoke(
        [
            "loops",
            "runs",
            "metrics",
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
    doc = _parse_jsonl(result.output)[0]
    assert doc["trainer_deployment_id"] == "dep_trainer"


def test_runs_metrics_handles_run_without_sampler(mock_remote):
    _setup_run_metrics_mocks(
        mock_remote,
        trainer_volume=[1.0, 2.0],
        trainer_concurrent=[1.0, 1.0],
        sampler_deployment_id=None,
    )
    result = _invoke(
        [
            "loops",
            "runs",
            "metrics",
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
    doc = _parse_jsonl(result.output)[0]
    assert doc["sampler_deployment_id"] is None
    assert doc["sampler"]["request_volume"] == []
    mock_remote.api.get_model_deployment_range_metrics.assert_not_called()


def test_runs_metrics_since_overrides_default_window(mock_remote):
    _setup_run_metrics_mocks(
        mock_remote, trainer_volume=[1.0], trainer_concurrent=[1.0]
    )
    result = _invoke(
        [
            "loops",
            "runs",
            "metrics",
            "--remote",
            "test_remote",
            "--run-id",
            "trnr_xyz",
            "--since",
            "1h",
            "-o",
            "json",
        ],
        mock_remote,
    )
    assert result.exit_code == 0, result.output
    # With --since 1h, the trainer-metrics call uses a start roughly one
    # hour before now (not run.created_at = 2026-05-07).
    call_kwargs = mock_remote.api.get_loops_deployment_metrics.call_args.kwargs
    assert call_kwargs["start_epoch_millis"] is not None
    # Sanity: start should be after the run's created_at of 2026-05-07.
    run_created_ms = 1746621296000  # 2026-05-07T12:34:56Z
    assert call_kwargs["start_epoch_millis"] > run_created_ms
