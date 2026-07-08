"""Tests for truss loops CLI commands."""

import json
import os
import re
from datetime import datetime
from unittest.mock import Mock, patch

import pytest
from click.testing import CliRunner

from truss.cli.cli import truss_cli
from truss.cli.logs.utils import (
    LOG_RESUME_FORMAT,
    MAX_LOG_RANGE,
    resolve_log_time_range,
)
from truss.cli.loops_commands import _logs_truncation_hint
from truss.remote.baseten.api import LoopsLogsWindow

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


def test_view_default_requests_caller_scope(mock_remote):
    mock_remote.api.list_loops_deployments.return_value = []
    result = _invoke(["loops", "view", "--remote", "test_remote"], mock_remote)
    assert result.exit_code == 0, result.output
    mock_remote.api.list_loops_deployments.assert_called_once_with(scope=None)


def test_view_org_flag_requests_org_scope(mock_remote):
    mock_remote.api.list_loops_deployments.return_value = []
    result = _invoke(["loops", "view", "--org", "--remote", "test_remote"], mock_remote)
    assert result.exit_code == 0, result.output
    mock_remote.api.list_loops_deployments.assert_called_once_with(scope="org")


def test_view_org_flag_renders_owner_column(mock_remote):
    mock_remote.api.list_loops_deployments.return_value = [
        {
            "id": "dep_abc",
            "base_model": "Qwen/Qwen3-8B",
            "base_url": "https://trainer-abc.api.baseten.co/trainer",
            "status": {"name": "RUNNING"},
            "user": {"email": "owner@baseten.co"},
            "sampler": None,
        }
    ]
    result = _invoke(["loops", "view", "--org", "--remote", "test_remote"], mock_remote)
    assert result.exit_code == 0, result.output
    assert "Owner" in result.output
    assert "owner@baseten.co" in result.output


def test_view_without_org_flag_hides_owner_column(mock_remote):
    mock_remote.api.list_loops_deployments.return_value = [
        {
            "id": "dep_abc",
            "base_model": "Qwen/Qwen3-8B",
            "base_url": "https://trainer-abc.api.baseten.co/trainer",
            "status": {"name": "RUNNING"},
            "user": {"email": "owner@baseten.co"},
            "sampler": None,
        }
    ]
    result = _invoke(["loops", "view", "--remote", "test_remote"], mock_remote)
    assert result.exit_code == 0, result.output
    assert "Owner" not in result.output
    assert "owner@baseten.co" not in result.output


def test_view_org_flag_owner_placeholder_when_user_missing(mock_remote):
    # Older backend with no ``user`` field: Owner degrades to a placeholder.
    mock_remote.api.list_loops_deployments.return_value = [
        {
            "id": "dep_abc",
            "base_model": "Qwen/Qwen3-8B",
            "base_url": "https://trainer-abc.api.baseten.co/trainer",
            "status": {"name": "RUNNING"},
            "sampler": None,
        }
    ]
    result = _invoke(["loops", "view", "--org", "--remote", "test_remote"], mock_remote)
    assert result.exit_code == 0, result.output
    assert "Owner" in result.output
    assert "dep_abc" in result.output


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


def test_logs_requires_exactly_one_deployment_id(mock_remote):
    result = _invoke(["loops", "logs", "--remote", "test_remote"], mock_remote)
    assert result.exit_code != 0
    assert "exactly one" in result.output

    result = _invoke(
        [
            "loops",
            "logs",
            "--remote",
            "test_remote",
            "--loops-deployment-id",
            "dep-1",
            "--sampler-deployment-id",
            "samp-1",
        ],
        mock_remote,
    )
    assert result.exit_code != 0
    assert "exactly one" in result.output


def test_logs_tail_rejects_time_range_flags(mock_remote):
    result = _invoke(
        [
            "loops",
            "logs",
            "--remote",
            "test_remote",
            "--loops-deployment-id",
            "dep-1",
            "--tail",
            "--since",
            "1h",
        ],
        mock_remote,
    )
    assert result.exit_code != 0
    assert "--tail cannot be combined" in result.output
    mock_remote.api.get_loops_deployment_logs.assert_not_called()


def test_logs_bare_invocation_uses_single_newest_page(mock_remote):
    # No window flags: one request for the newest server page, no
    # pagination, no hint.
    mock_remote.api.get_loops_deployment_logs_page.return_value = [
        {"timestamp": "1700000000000000000", "message": "hello world", "replica": None}
    ]
    result = _invoke(
        ["loops", "logs", "--remote", "test_remote", "--loops-deployment-id", "dep-1"],
        mock_remote,
    )
    assert result.exit_code == 0, result.output
    mock_remote.api.get_loops_deployment_logs_page.assert_called_once_with("dep-1")
    mock_remote.api.get_loops_deployment_logs.assert_not_called()
    assert "hello world" in result.output
    assert "Showing the first" not in result.output


def test_logs_since_passes_resolved_time_range(mock_remote):
    mock_remote.api.get_loops_deployment_logs.return_value = LoopsLogsWindow(
        logs=[], truncated=False, resume_start_epoch_millis=None
    )
    result = _invoke(
        [
            "loops",
            "logs",
            "--remote",
            "test_remote",
            "--loops-deployment-id",
            "dep-1",
            "--since",
            "1h",
        ],
        mock_remote,
    )
    assert result.exit_code == 0, result.output
    _, start_ms, end_ms = mock_remote.api.get_loops_deployment_logs.call_args[0]
    assert end_ms - start_ms == 3600 * 1000


def test_logs_sampler_path_passes_time_range(mock_remote):
    mock_remote.api.list_loops_deployments.return_value = [
        {"sampler": {"deployment_id": "samp-1", "model_id": "model-9"}}
    ]
    mock_remote.api.get_model_deployment_logs.return_value = []
    result = _invoke(
        [
            "loops",
            "logs",
            "--remote",
            "test_remote",
            "--sampler-deployment-id",
            "samp-1",
            "--since",
            "2h",
        ],
        mock_remote,
    )
    assert result.exit_code == 0, result.output
    model_id, deployment_id, start_ms, end_ms = (
        mock_remote.api.get_model_deployment_logs.call_args[0]
    )
    assert model_id == "model-9"
    assert deployment_id == "samp-1"
    assert end_ms - start_ms == 2 * 3600 * 1000


def test_logs_limit_truncation_prints_resume_hint(mock_remote):
    # The pager reports truncation and the exact resume cursor; the hint
    # renders it as a ready-to-paste --start value.
    mock_remote.api.get_loops_deployment_logs.return_value = LoopsLogsWindow(
        logs=[
            {"timestamp": "1700000000000000000", "message": "one", "replica": None},
            {"timestamp": "1700000060000000000", "message": "two", "replica": None},
        ],
        truncated=True,
        resume_start_epoch_millis=1700000120000,
    )
    result = _invoke(
        [
            "loops",
            "logs",
            "--remote",
            "test_remote",
            "--loops-deployment-id",
            "dep-1",
            "--limit",
            "2",
        ],
        mock_remote,
    )
    assert result.exit_code == 0, result.output
    mock_remote.api.get_loops_deployment_logs.assert_called_once_with(
        "dep-1", None, None, max_lines=2
    )
    assert "two" in result.output
    assert "Showing the first 2 lines" in result.output
    assert "--start" in result.output
    # No explicit window end was given, so the hint must not suggest --end.
    assert "--end" not in result.output


def test_logs_exact_fit_prints_no_hint(mock_remote):
    # A window holding exactly --limit lines comes back not truncated: no
    # hint.
    mock_remote.api.get_loops_deployment_logs.return_value = LoopsLogsWindow(
        logs=[
            {"timestamp": "1700000000000000000", "message": "one", "replica": None},
            {"timestamp": "1700000060000000000", "message": "two", "replica": None},
        ],
        truncated=False,
        resume_start_epoch_millis=None,
    )
    result = _invoke(
        [
            "loops",
            "logs",
            "--remote",
            "test_remote",
            "--loops-deployment-id",
            "dep-1",
            "--limit",
            "2",
        ],
        mock_remote,
    )
    assert result.exit_code == 0, result.output
    assert "Showing the first" not in result.output


def test_logs_truncation_hint_preserves_window_end(mock_remote):
    # With an explicit window (--since resolves an end), the resume hint must
    # carry --end so the resumed fetch stays inside the original window.
    mock_remote.api.get_loops_deployment_logs.return_value = LoopsLogsWindow(
        logs=[
            {"timestamp": "1700000000000000000", "message": "one", "replica": None},
            {"timestamp": "1700000060000000000", "message": "two", "replica": None},
        ],
        truncated=True,
        resume_start_epoch_millis=1700000120000,
    )
    result = _invoke(
        [
            "loops",
            "logs",
            "--remote",
            "test_remote",
            "--loops-deployment-id",
            "dep-1",
            "--since",
            "1h",
            "--limit",
            "2",
        ],
        mock_remote,
    )
    assert result.exit_code == 0, result.output
    assert "Showing the first 2 lines" in result.output
    assert '--end "' in result.output


def test_logs_stuck_millisecond_hint_omits_unusable_resume(mock_remote):
    # The pager reports truncation with no usable cursor (one millisecond
    # holds more lines than --limit); the hint must not offer a resume.
    mock_remote.api.get_loops_deployment_logs.return_value = LoopsLogsWindow(
        logs=[
            {"timestamp": "1700000000000000100", "message": "a", "replica": None},
            {"timestamp": "1700000000000000200", "message": "b", "replica": None},
        ],
        truncated=True,
        resume_start_epoch_millis=None,
    )
    result = _invoke(
        [
            "loops",
            "logs",
            "--remote",
            "test_remote",
            "--loops-deployment-id",
            "dep-1",
            "--limit",
            "2",
        ],
        mock_remote,
    )
    assert result.exit_code == 0, result.output
    assert "share one millisecond" in result.output
    assert "--start" not in result.output.replace("--start cannot", "")
    assert "raise --limit" in result.output


@patch("truss.cli.loops_commands.LOOPS_LOGS_MAX_LINES", 2)
def test_logs_hint_drops_raise_limit_clause_at_ceiling(mock_remote):
    mock_remote.api.get_loops_deployment_logs.return_value = LoopsLogsWindow(
        logs=[
            {"timestamp": "1700000000000000000", "message": "one", "replica": None},
            {"timestamp": "1700000060000000000", "message": "two", "replica": None},
        ],
        truncated=True,
        resume_start_epoch_millis=1700000120000,
    )
    result = _invoke(
        [
            "loops",
            "logs",
            "--remote",
            "test_remote",
            "--loops-deployment-id",
            "dep-1",
            "--limit",
            "2",
        ],
        mock_remote,
    )
    assert result.exit_code == 0, result.output
    assert "Showing the first 2 lines" in result.output
    assert "raise --limit" not in result.output


def test_logs_limit_rejected_with_tail(mock_remote):
    result = _invoke(
        [
            "loops",
            "logs",
            "--remote",
            "test_remote",
            "--loops-deployment-id",
            "dep-1",
            "--tail",
            "--limit",
            "100",
        ],
        mock_remote,
    )
    assert result.exit_code != 0
    assert "--tail cannot be combined" in result.output


def test_logs_limit_rejected_with_sampler(mock_remote):
    result = _invoke(
        [
            "loops",
            "logs",
            "--remote",
            "test_remote",
            "--sampler-deployment-id",
            "samp-1",
            "--limit",
            "100",
        ],
        mock_remote,
    )
    assert result.exit_code != 0
    assert "--limit only applies" in result.output


def test_logs_truncation_hint_command_survives_seven_day_check():
    # Pasting the hint's --start/--end back must never fail window
    # validation, even when the original window was exactly 7 days and the
    # resume cursor sits at its very start (the --end bias would otherwise
    # widen the span past the limit).
    end_ms = 1782968142946
    resume_ms = end_ms - int(MAX_LOG_RANGE.total_seconds() * 1000)
    hint = _logs_truncation_hint(resume_ms, max_lines=3, end_ms=end_ms)

    start_str = hint.split('--start "')[1].split('"')[0]
    end_str = hint.split('--end "')[1].split('"')[0]
    start_dt = datetime.strptime(start_str, LOG_RESUME_FORMAT)
    end_dt = datetime.strptime(end_str, LOG_RESUME_FORMAT)
    _, resolved_end = resolve_log_time_range(start_dt, end_dt, None)  # no raise
    assert resolved_end >= end_ms
