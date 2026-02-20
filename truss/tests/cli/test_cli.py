import threading
from unittest.mock import MagicMock, Mock, patch

import pytest
import requests
from click.testing import CliRunner

from truss.cli.cli import truss_cli
from truss.cli.utils import common
from truss.remote.baseten.custom_types import OidcInfo, OidcTeamInfo
from truss.remote.truss_remote import RemoteUser


def test_push_with_grpc_transport_fails_for_development_deployment():
    mock_truss = Mock()
    mock_truss.spec.config.runtime.transport.kind = "grpc"

    runner = CliRunner()

    with patch("truss.cli.cli._get_truss_from_directory", return_value=mock_truss):
        with patch("truss.cli.remote_cli.inquire_remote_name", return_value="remote1"):
            result = runner.invoke(
                truss_cli,
                [
                    "push",
                    "test_truss",
                    "--remote",
                    "remote1",
                    "--model-name",
                    "name",
                    "--watch",
                ],
            )

    assert result.exit_code == 2
    assert (
        "Truss with gRPC transport cannot be used as a development deployment"
        in result.output
    )


# keepalive_loop tests


def test_successful_ping_resets_failure_count():
    """A 200 response should reset consecutive failures to 0."""
    stop_event = threading.Event()
    call_count = 0

    def mock_get(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        resp = Mock()
        if call_count <= 2:
            # First two calls fail
            resp.status_code = 500
            resp.json.return_value = {}
        elif call_count == 3:
            # Third call succeeds - should reset counter
            resp.status_code = 200
        else:
            # Fourth call: stop
            stop_event.set()
            resp.status_code = 200
        return resp

    with patch("truss.cli.cli.requests_lib.get", side_effect=mock_get):
        with patch("truss.cli.cli.console"):
            # Use a very short wait so the test runs fast
            with patch.object(stop_event, "wait", side_effect=lambda timeout: None):
                common.keepalive_loop(
                    "http://fake/development/sync/v1/models/model",
                    "test_api_key",
                    stop_event,
                )
    assert call_count == 4  # All calls were made, no early exit


def test_exits_after_max_consecutive_failures():
    """Should call os._exit(1) after max consecutive failures."""
    stop_event = threading.Event()
    mock_resp = Mock()
    mock_resp.status_code = 500
    mock_resp.json.return_value = {}
    with patch("truss.cli.cli.requests_lib.get", return_value=mock_resp):
        with patch("truss.cli.cli.console") as _mock_console:
            with patch(
                "truss.cli.cli.os._exit", side_effect=lambda code: stop_event.set()
            ) as mock_exit:
                with patch.object(stop_event, "wait", side_effect=lambda timeout: None):
                    common.keepalive_loop(
                        "http://fake/development/sync/v1/models/model",
                        "test_api_key",
                        stop_event,
                    )
    mock_exit.assert_called_once_with(1)


def test_request_exception_counts_as_failure():
    """Network errors should count toward consecutive failures."""
    stop_event = threading.Event()

    with patch(
        "truss.cli.cli.requests_lib.get",
        side_effect=requests.RequestException("connection error"),
    ):
        with patch("truss.cli.cli.console"):
            with patch(
                "truss.cli.cli.os._exit", side_effect=lambda code: stop_event.set()
            ) as mock_exit:
                with patch.object(stop_event, "wait", side_effect=lambda timeout: None):
                    common.keepalive_loop(
                        "http://fake/development/sync/v1/models/model",
                        "test_api_key",
                        stop_event,
                    )

    mock_exit.assert_called_once_with(1)


def test_model_not_ready_does_not_count_as_failure():
    """'Model is not ready' errors during patching should be ignored."""
    stop_event = threading.Event()
    call_count = 0

    def mock_get(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        resp = Mock()
        if call_count <= 25:
            resp.status_code = 400
            resp.json.return_value = {
                "error": "Model is not ready, it is still building or deploying"
            }
        else:
            stop_event.set()
            resp.status_code = 200
        return resp

    with patch("truss.cli.cli.requests_lib.get", side_effect=mock_get):
        with patch("truss.cli.cli.console"):
            with patch(
                "truss.cli.cli.os._exit", side_effect=lambda code: stop_event.set()
            ) as mock_exit:
                with patch.object(stop_event, "wait", side_effect=lambda timeout: None):
                    common.keepalive_loop(
                        "http://fake/development/sync/v1/models/model",
                        "test_api_key",
                        stop_event,
                    )

    mock_exit.assert_not_called()


def test_keepalive_loop_emits_30min_warning():
    """Test that keepalive loop emits a warning 30 minutes before the 24-hour exit."""
    stop_event = threading.Event()

    # Mock time.time() to simulate being just past 23.5 hours (to trigger warning)
    time_23_5_hours_plus = (23.5 * 60 * 60) + 1  # 84601 seconds

    iteration_count = [0]

    def mock_time():
        """First call returns 0 (start_time), subsequent calls return 23.5+ hours"""
        if iteration_count[0] == 0:
            iteration_count[0] += 1
            return 0.0
        return time_23_5_hours_plus

    def mock_wait(timeout=None):
        """Stop after first iteration"""
        stop_event.set()
        return False

    with patch("truss.cli.utils.common.time.time", side_effect=mock_time):
        with patch("truss.cli.utils.common.requests_lib") as mock_requests:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_requests.get.return_value = mock_response
            mock_requests.RequestException = requests.RequestException

            with patch("truss.cli.utils.common.console.print") as mock_console_print:
                # Patch the wait method to stop after first call
                stop_event.wait = mock_wait

                common.keepalive_loop(
                    model_hostname="https://test.baseten.co",
                    api_key="test-key",
                    stop_event=stop_event,
                )

                # Verify the warning was printed
                mock_console_print.assert_any_call(
                    "⚠️  Keepalive will automatically exit in 30 minutes (24 hour limit).",
                    style="yellow",
                )


# truss watch --no-sleep tests


def _make_watch_mocks(
    model_hostname="https://model-abc123.api.baseten.co", hostname_present=True
):
    """Helper to create common mocks for watch --no-sleep tests."""
    resolved_model = {
        "id": "model_id",
        "name": "test_model",
        "hostname": model_hostname if hostname_present else None,
        "versions": [{"id": "dev_version_id", "is_draft": True, "is_primary": False}],
    }
    versions = resolved_model["versions"]
    dev_version = versions[0]

    mock_tr = Mock()
    mock_tr.spec.config.model_name = "test_model"

    remote_provider = MagicMock()
    remote_provider._auth_service.authenticate.return_value = Mock(value="test_key")
    remote_provider.remote_url = "https://app.baseten.co"
    remote_provider.api.get_deployment.return_value = {"status": "ACTIVE"}

    return resolved_model, versions, dev_version, mock_tr, remote_provider


def _patch_watch_common(
    remote_provider, mock_tr, resolved_model, versions, dev_version
):
    """Returns a contextmanager-like stack of patches for watch tests."""
    from contextlib import ExitStack

    stack = ExitStack()
    stack.enter_context(
        patch("truss.cli.cli.RemoteFactory.create", return_value=remote_provider)
    )
    stack.enter_context(
        patch("truss.cli.cli._get_truss_from_directory", return_value=mock_tr)
    )
    stack.enter_context(
        patch(
            "truss.cli.cli.resolve_model_for_watch",
            return_value=(resolved_model, versions),
        )
    )
    stack.enter_context(
        patch("truss.cli.cli.get_dev_version_from_versions", return_value=dev_version)
    )
    stack.enter_context(
        patch("truss.cli.cli.RemoteFactory.get_remote_team", return_value=None)
    )
    stack.enter_context(patch("truss.cli.cli.time.sleep"))
    return stack


def test_watch_sends_wake_request():
    """Watch command should POST to /development/wake before waiting (regardless of --no-sleep)."""
    resolved_model, versions, dev_version, mock_tr, remote_provider = (
        _make_watch_mocks()
    )
    runner = CliRunner()

    with _patch_watch_common(
        remote_provider, mock_tr, resolved_model, versions, dev_version
    ):
        # Need to patch requests_lib in common.py now since wake is called from there
        with patch("truss.cli.utils.common.requests_lib") as mock_requests:
            mock_requests.post.return_value = Mock(status_code=202)
            mock_requests.RequestException = requests.RequestException
            with patch.object(remote_provider, "sync_truss_to_dev_version_with_model"):
                _result = runner.invoke(
                    truss_cli,
                    ["watch", "/tmp/fake", "--remote", "baseten"],  # No --no-sleep
                )

    mock_requests.post.assert_called_once_with(
        "https://model-abc123.api.baseten.co/development/wake",
        headers={"Authorization": "Api-Key test_key"},
        timeout=10,
    )


def test_watch_no_sleep_starts_keepalive_thread():
    """--no-sleep should start a daemon keepalive thread after model is ready."""
    resolved_model, versions, dev_version, mock_tr, remote_provider = (
        _make_watch_mocks()
    )
    runner = CliRunner()

    with _patch_watch_common(
        remote_provider, mock_tr, resolved_model, versions, dev_version
    ):
        with patch("truss.cli.utils.common.requests_lib") as mock_requests:
            mock_requests.post.return_value = Mock(status_code=202)
            mock_requests.get.return_value = Mock(status_code=200)
            mock_requests.RequestException = requests.RequestException
            with patch("truss.cli.cli.threading.Thread") as mock_thread_cls:
                mock_thread = Mock()
                mock_thread_cls.return_value = mock_thread
                with patch.object(
                    remote_provider, "sync_truss_to_dev_version_with_model"
                ):
                    _result = runner.invoke(
                        truss_cli,
                        ["watch", "/tmp/fake", "--remote", "baseten", "--no-sleep"],
                    )

    mock_thread_cls.assert_called_once()
    _, kwargs = mock_thread_cls.call_args
    assert kwargs["daemon"] is True
    assert kwargs["target"] == common.keepalive_loop
    thread_args = kwargs["args"]
    assert thread_args[0] == "https://model-abc123.api.baseten.co"
    assert thread_args[1] == "test_key"
    mock_thread.start.assert_called_once()


def test_keepalive_loop_constructs_correct_url():
    """Keepalive loop should construct the correct health check URL from hostname."""
    stop_event = threading.Event()

    with patch("truss.cli.utils.common.time.time") as mock_time:
        # First call: start_time, subsequent calls: still within duration
        mock_time.side_effect = [0.0, 100.0, 200.0]

        with patch("truss.cli.utils.common.requests_lib") as mock_requests:
            mock_requests.RequestException = requests.RequestException
            mock_response = Mock(status_code=200)
            mock_requests.get.return_value = mock_response

            # Stop after first iteration
            def stop_after_first(*args, **kwargs):
                stop_event.set()
                return False

            with patch.object(stop_event, "wait", side_effect=stop_after_first):
                common.keepalive_loop(
                    model_hostname="https://model-abc123.api.baseten.co",
                    api_key="test-key",
                    stop_event=stop_event,
                )

            # Verify the URL was constructed correctly
            mock_requests.get.assert_called_once_with(
                "https://model-abc123.api.baseten.co/development/sync/v1/models/model",
                headers={"Authorization": "Api-Key test-key"},
                timeout=10,
            )


def test_watch_no_sleep_waits_for_active_status():
    """--no-sleep should poll deployment status until ACTIVE."""
    resolved_model, versions, dev_version, mock_tr, remote_provider = (
        _make_watch_mocks()
    )
    remote_provider.api.get_deployment.side_effect = [
        {"status": "SCALED_TO_ZERO"},
        {"status": "WAKING_UP"},
        {"status": "ACTIVE"},
    ]
    runner = CliRunner()

    with _patch_watch_common(
        remote_provider, mock_tr, resolved_model, versions, dev_version
    ):
        with patch("truss.cli.cli.requests_lib") as mock_requests:
            mock_requests.post.return_value = Mock(status_code=202)
            mock_requests.get.return_value = Mock(status_code=200)
            mock_requests.RequestException = requests.RequestException
            with patch("truss.cli.cli.threading.Thread") as mock_thread_cls:
                mock_thread_cls.return_value = Mock()
                with patch.object(
                    remote_provider, "sync_truss_to_dev_version_with_model"
                ):
                    result = runner.invoke(
                        truss_cli,
                        ["watch", "/tmp/fake", "--remote", "baseten", "--no-sleep"],
                    )

    assert result.exit_code == 0
    assert remote_provider.api.get_deployment.call_count == 3


def test_watch_no_sleep_exits_on_failed_deployment():
    """--no-sleep should exit if deployment reaches a terminal failure status."""
    resolved_model, versions, dev_version, mock_tr, remote_provider = (
        _make_watch_mocks()
    )
    remote_provider.api.get_deployment.return_value = {"status": "FAILED"}
    runner = CliRunner()

    with _patch_watch_common(
        remote_provider, mock_tr, resolved_model, versions, dev_version
    ):
        with patch("truss.cli.cli.requests_lib") as mock_requests:
            mock_requests.post.return_value = Mock(status_code=202)
            mock_requests.RequestException = requests.RequestException
            result = runner.invoke(
                truss_cli, ["watch", "/tmp/fake", "--remote", "baseten", "--no-sleep"]
            )

    assert result.exit_code != 0
    assert "Development model failed with status FAILED" in result.output


def test_watch_without_no_sleep_does_not_start_thread():
    """Without --no-sleep, no keepalive thread should be started."""
    resolved_model, versions, dev_version, mock_tr, remote_provider = (
        _make_watch_mocks()
    )
    runner = CliRunner()

    with _patch_watch_common(
        remote_provider, mock_tr, resolved_model, versions, dev_version
    ):
        with patch("truss.cli.cli.threading.Thread") as mock_thread_cls:
            with patch.object(remote_provider, "sync_truss_to_dev_version_with_model"):
                _result = runner.invoke(
                    truss_cli, ["watch", "/tmp/fake", "--remote", "baseten"]
                )

    mock_thread_cls.assert_not_called()


def test_keepalive_loop_continues_before_max_duration():
    """Keepalive loop should keep running before 24 hours."""
    stop_event = threading.Event()

    with patch("truss.cli.utils.common.requests_lib") as mock_requests:
        mock_requests.get.return_value = Mock(status_code=200)
        mock_requests.RequestException = requests.RequestException

        with patch("truss.cli.utils.common.time.time") as mock_time:
            mock_time.side_effect = [
                0,  # start_time = 0
                100,  # iteration 1: elapsed_time check
                100,  # iteration 1: max duration check (100 < 86400 → continue)
                200,  # iteration 2: elapsed_time check
                200,  # iteration 2: max duration check (200 < 86400 → continue)
                86401,  # iteration 3: elapsed_time check
                86401,  # iteration 3: max duration check (86401 > 86400 → exit)
            ]
            with patch(
                "truss.cli.utils.common.os._exit", side_effect=SystemExit(0)
            ) as mock_exit:
                with patch.object(stop_event, "wait"):
                    with pytest.raises(SystemExit):
                        common.keepalive_loop(
                            "https://model-abc123.api.baseten.co",
                            "fake_key",
                            stop_event,
                        )

    assert mock_requests.get.call_count == 2
    mock_exit.assert_called_once_with(0)


def test_cli_push_passes_deploy_timeout_minutes_to_create_truss_service(
    custom_model_truss_dir_with_pre_and_post,
    remote,
    mock_baseten_requests,
    mock_upload_truss,
    mock_create_truss_service,
):
    runner = CliRunner()
    with patch("truss.cli.cli.RemoteFactory.create", return_value=remote):
        remote.api.get_teams = Mock(return_value={})
        with patch("truss.cli.cli.resolve_model_team_name", return_value=(None, None)):
            result = runner.invoke(
                truss_cli,
                [
                    "push",
                    str(custom_model_truss_dir_with_pre_and_post),
                    "--remote",
                    "baseten",
                    "--model-name",
                    "model_name",
                    "--deploy-timeout-minutes",
                    "450",
                ],
            )

    assert result.exit_code == 0
    mock_create_truss_service.assert_called_once()
    _, kwargs = mock_create_truss_service.call_args
    assert kwargs["deploy_timeout_minutes"] == 450


def test_cli_push_passes_none_deploy_timeout_minutes_when_not_specified(
    custom_model_truss_dir_with_pre_and_post,
    remote,
    mock_baseten_requests,
    mock_upload_truss,
    mock_create_truss_service,
):
    runner = CliRunner()
    with patch("truss.cli.cli.RemoteFactory.create", return_value=remote):
        remote.api.get_teams = Mock(return_value={})
        with patch("truss.cli.cli.resolve_model_team_name", return_value=(None, None)):
            result = runner.invoke(
                truss_cli,
                [
                    "push",
                    str(custom_model_truss_dir_with_pre_and_post),
                    "--remote",
                    "baseten",
                    "--model-name",
                    "model_name",
                ],
            )

    assert result.exit_code == 0
    mock_create_truss_service.assert_called_once()
    _, kwargs = mock_create_truss_service.call_args
    assert kwargs.get("deploy_timeout_minutes") is None


def test_cli_push_integration_deploy_timeout_minutes_propagated(
    custom_model_truss_dir_with_pre_and_post,
    remote,
    mock_baseten_requests,
    mock_upload_truss,
    mock_create_truss_service,
):
    runner = CliRunner()
    with patch("truss.cli.cli.RemoteFactory.create", return_value=remote):
        remote.api.get_teams = Mock(return_value={})
        with patch("truss.cli.cli.resolve_model_team_name", return_value=(None, None)):
            result = runner.invoke(
                truss_cli,
                [
                    "push",
                    str(custom_model_truss_dir_with_pre_and_post),
                    "--remote",
                    "baseten",
                    "--model-name",
                    "model_name",
                    "--environment",
                    "staging",
                    "--deploy-timeout-minutes",
                    "750",
                ],
            )

    assert result.exit_code == 0
    mock_create_truss_service.assert_called_once()
    _, kwargs = mock_create_truss_service.call_args
    assert kwargs["deploy_timeout_minutes"] == 750
    assert kwargs["environment"] == "staging"


def test_cli_push_api_integration_deploy_timeout_minutes_propagated(
    custom_model_truss_dir_with_pre_and_post,
    mock_remote_factory,
    temp_trussrc_dir,
    mock_available_config_names,
):
    mock_service = MagicMock()
    mock_service.model_id = "model_id"
    mock_service.model_version_id = "version_id"
    mock_remote_factory.push.return_value = mock_service

    runner = CliRunner()
    with patch(
        "truss.cli.cli.RemoteFactory.get_available_config_names",
        return_value=["baseten"],
    ):
        result = runner.invoke(
            truss_cli,
            [
                "push",
                str(custom_model_truss_dir_with_pre_and_post),
                "--remote",
                "baseten",
                "--model-name",
                "test_model",
                "--deploy-timeout-minutes",
                "1200",
            ],
        )

    assert result.exit_code == 0
    mock_remote_factory.push.assert_called_once()
    _, push_kwargs = mock_remote_factory.push.call_args
    assert push_kwargs.get("deploy_timeout_minutes") == 1200


def test_whoami_basic():
    """Test basic whoami command."""
    runner = CliRunner()

    mock_user = RemoteUser("test_workspace", "user@example.com")

    with patch("truss.cli.remote_cli.inquire_remote_name", return_value="baseten"):
        with patch("truss.api.whoami", return_value=mock_user) as mock_whoami_fn:
            result = runner.invoke(truss_cli, ["whoami", "--remote", "baseten"])

    assert result.exit_code == 0
    assert "test_workspace\\user@example.com" in result.output
    mock_whoami_fn.assert_called_once_with("baseten")


def test_whoami_with_show_oidc():
    """Test whoami command with --show-oidc flag displays OIDC information."""
    runner = CliRunner()

    mock_user = RemoteUser("test_workspace", "user@example.com")
    mock_oidc_info = OidcInfo(
        org_id="PJAd5Q0",
        teams=[
            OidcTeamInfo(id="wgeyxoq", name="Default Team"),
            OidcTeamInfo(id="abc123", name="ML Team"),
        ],
        issuer="https://oidc.baseten.co",
        audience="oidc.baseten.co",
        workload_types=["model_container", "model_build"],
    )

    mock_remote = MagicMock()
    mock_remote.get_oidc_info.return_value = mock_oidc_info

    with patch("truss.cli.remote_cli.inquire_remote_name", return_value="baseten"):
        with patch("truss.api.whoami", return_value=mock_user):
            with patch("truss.cli.cli.RemoteFactory.create", return_value=mock_remote):
                result = runner.invoke(
                    truss_cli, ["whoami", "--remote", "baseten", "--show-oidc"]
                )

    assert result.exit_code == 0
    assert "test_workspace\\user@example.com" in result.output
    assert "OIDC Configuration for Workload Identity" in result.output
    assert "PJAd5Q0" in result.output
    assert "wgeyxoq (Default Team)" in result.output
    assert "abc123 (ML Team)" in result.output
    assert "https://oidc.baseten.co" in result.output
    assert "oidc.baseten.co" in result.output


def test_push_defaults_to_published(
    custom_model_truss_dir_with_pre_and_post,
    remote,
    mock_baseten_requests,
    mock_upload_truss,
    mock_create_truss_service,
):
    """Test that push defaults to published deployment (is_draft=False)."""
    runner = CliRunner()
    with patch("truss.cli.cli.RemoteFactory.create", return_value=remote):
        remote.api.get_teams = Mock(return_value={})
        with patch("truss.cli.cli.resolve_model_team_name", return_value=(None, None)):
            result = runner.invoke(
                truss_cli,
                [
                    "push",
                    str(custom_model_truss_dir_with_pre_and_post),
                    "--remote",
                    "baseten",
                    "--model-name",
                    "model_name",
                ],
            )

    assert result.exit_code == 0
    mock_create_truss_service.assert_called_once()
    _, kwargs = mock_create_truss_service.call_args
    assert kwargs["is_draft"] is False


def test_push_publish_flag_shows_deprecation_warning(
    custom_model_truss_dir_with_pre_and_post,
    remote,
    mock_baseten_requests,
    mock_upload_truss,
    mock_create_truss_service,
):
    """Test that --publish flag shows deprecation warning."""
    runner = CliRunner()
    with patch("truss.cli.cli.RemoteFactory.create", return_value=remote):
        remote.api.get_teams = Mock(return_value={})
        with patch("truss.cli.cli.resolve_model_team_name", return_value=(None, None)):
            result = runner.invoke(
                truss_cli,
                [
                    "push",
                    str(custom_model_truss_dir_with_pre_and_post),
                    "--remote",
                    "baseten",
                    "--model-name",
                    "model_name",
                    "--publish",
                ],
            )

    assert result.exit_code == 0
    assert "DEPRECATED" in result.output


def test_push_watch_creates_development_deployment(
    custom_model_truss_dir_with_pre_and_post,
    remote,
    mock_baseten_requests,
    mock_upload_truss,
    mock_create_truss_service,
):
    """Test that --watch creates a development deployment (is_draft=True)."""
    runner = CliRunner()

    # Mock the service to return is_draft=True and have poll_deployment_status
    mock_service = MagicMock()
    mock_service.is_draft = True
    mock_service.logs_url = "https://example.com/logs"
    mock_service.poll_deployment_status.return_value = iter(["MODEL_READY"])
    remote.push = Mock(return_value=mock_service)

    with patch("truss.cli.cli.RemoteFactory.create", return_value=remote):
        remote.api.get_teams = Mock(return_value={})
        with patch("truss.cli.cli.resolve_model_team_name", return_value=(None, None)):
            with patch.object(remote, "sync_truss_to_dev_version_by_name"):
                _result = runner.invoke(
                    truss_cli,
                    [
                        "push",
                        str(custom_model_truss_dir_with_pre_and_post),
                        "--remote",
                        "baseten",
                        "--model-name",
                        "model_name",
                        "--watch",
                    ],
                )

    # Check push was called with publish=False
    remote.push.assert_called_once()
    _, kwargs = remote.push.call_args
    assert kwargs["publish"] is False


def test_push_watch_with_promote_fails():
    """Test that --watch with --promote fails."""
    mock_truss = Mock()
    mock_truss.spec.config.runtime.transport.kind = "http"
    mock_truss.spec.config.resources.instance_type = None
    mock_truss.spec.config.build = None
    mock_truss.spec.config.trt_llm = None

    runner = CliRunner()

    with patch("truss.cli.cli._get_truss_from_directory", return_value=mock_truss):
        result = runner.invoke(
            truss_cli,
            [
                "push",
                "test_truss",
                "--remote",
                "remote1",
                "--model-name",
                "name",
                "--watch",
                "--promote",
            ],
        )

    assert result.exit_code == 2
    assert "Cannot use --watch with --promote" in result.output or (
        result.exception
        and "Cannot use --watch with --promote" in str(result.exception.__context__)
    )


def test_push_watch_with_environment_fails():
    """Test that --watch with --environment fails."""
    mock_truss = Mock()
    mock_truss.spec.config.runtime.transport.kind = "http"
    mock_truss.spec.config.resources.instance_type = None
    mock_truss.spec.config.build = None
    mock_truss.spec.config.trt_llm = None

    runner = CliRunner()

    with patch("truss.cli.cli._get_truss_from_directory", return_value=mock_truss):
        result = runner.invoke(
            truss_cli,
            [
                "push",
                "test_truss",
                "--remote",
                "remote1",
                "--model-name",
                "name",
                "--watch",
                "--environment",
                "staging",
            ],
        )

    assert result.exit_code == 2
    assert "Cannot use --watch with --environment" in result.output or (
        result.exception
        and "Cannot use --watch with --environment" in str(result.exception.__context__)
    )


def test_push_watch_with_tail_fails():
    """Test that --watch with --tail fails."""
    mock_truss = Mock()
    mock_truss.spec.config.runtime.transport.kind = "http"
    mock_truss.spec.config.resources.instance_type = None
    mock_truss.spec.config.build = None
    mock_truss.spec.config.trt_llm = None

    runner = CliRunner()

    with patch("truss.cli.cli._get_truss_from_directory", return_value=mock_truss):
        result = runner.invoke(
            truss_cli,
            [
                "push",
                "test_truss",
                "--remote",
                "remote1",
                "--model-name",
                "name",
                "--watch",
                "--tail",
            ],
        )

    assert result.exit_code == 2
    assert "Cannot use --watch with --tail" in result.output or (
        result.exception
        and "Cannot use --watch with --tail" in str(result.exception.__context__)
    )


def test_push_watch_with_publish_fails():
    """Test that --watch with --publish fails."""
    mock_truss = Mock()
    mock_truss.spec.config.runtime.transport.kind = "http"
    mock_truss.spec.config.resources.instance_type = None
    mock_truss.spec.config.build = None
    mock_truss.spec.config.trt_llm = None

    runner = CliRunner()

    with patch("truss.cli.cli._get_truss_from_directory", return_value=mock_truss):
        result = runner.invoke(
            truss_cli,
            [
                "push",
                "test_truss",
                "--remote",
                "remote1",
                "--model-name",
                "name",
                "--watch",
                "--publish",
            ],
        )

    assert result.exit_code == 2
    assert "Cannot use --watch with --publish" in result.output or (
        result.exception
        and "Cannot use --watch with --publish" in str(result.exception.__context__)
    )


def test_push_wait_with_tail_fails():
    """Test that --wait with --tail fails."""
    mock_truss = Mock()
    mock_truss.spec.config.runtime.transport.kind = "http"
    mock_truss.spec.config.resources.instance_type = None
    mock_truss.spec.config.build = None
    mock_truss.spec.config.trt_llm = None

    runner = CliRunner()

    with patch("truss.cli.cli._get_truss_from_directory", return_value=mock_truss):
        result = runner.invoke(
            truss_cli,
            [
                "push",
                "test_truss",
                "--remote",
                "remote1",
                "--model-name",
                "name",
                "--wait",
                "--tail",
            ],
        )

    assert result.exit_code == 2
    assert "Cannot use --wait with --tail" in result.output or (
        result.exception
        and "Cannot use --wait with --tail" in str(result.exception.__context__)
    )


def test_push_watch_enters_watch_mode_on_deploying_status(
    custom_model_truss_dir_with_pre_and_post,
    remote,
    mock_baseten_requests,
    mock_upload_truss,
    mock_create_truss_service,
):
    """Test that --watch enters watch mode early when status is LOADING_MODEL,
    without waiting for ACTIVE."""
    runner = CliRunner()

    mock_service = MagicMock()
    mock_service.is_draft = True
    mock_service.logs_url = "https://example.com/logs"
    # Simulate: BUILDING -> LOADING_MODEL — never reaches ACTIVE
    mock_service.poll_deployment_status.return_value = iter(
        ["BUILDING", "LOADING_MODEL"]
    )
    remote.push = Mock(return_value=mock_service)

    mock_resolve = Mock(
        return_value=(
            {"id": "model_id", "name": "model_name"},
            [{"id": "version_id", "is_draft": True}],
        )
    )
    mock_start_watch = Mock()

    with patch("truss.cli.cli.RemoteFactory.create", return_value=remote):
        remote.api.get_teams = Mock(return_value={})
        with patch("truss.cli.cli.resolve_model_team_name", return_value=(None, None)):
            with patch("truss.cli.cli.resolve_model_for_watch", mock_resolve):
                with patch("truss.cli.cli._start_watch_mode", mock_start_watch):
                    result = runner.invoke(
                        truss_cli,
                        [
                            "push",
                            str(custom_model_truss_dir_with_pre_and_post),
                            "--remote",
                            "baseten",
                            "--model-name",
                            "model_name",
                            "--watch",
                        ],
                    )

    assert result.exit_code == 0
    mock_start_watch.assert_called_once()


def test_push_watch_still_exits_on_deploy_failed(
    custom_model_truss_dir_with_pre_and_post,
    remote,
    mock_baseten_requests,
    mock_upload_truss,
    mock_create_truss_service,
):
    """Test that --watch still sys.exit(1) on a real failure like DEPLOY_FAILED."""
    runner = CliRunner()

    mock_service = MagicMock()
    mock_service.is_draft = True
    mock_service.logs_url = "https://example.com/logs"
    mock_service.poll_deployment_status.return_value = iter(
        ["BUILDING", "DEPLOY_FAILED"]
    )
    remote.push = Mock(return_value=mock_service)

    with patch("truss.cli.cli.RemoteFactory.create", return_value=remote):
        remote.api.get_teams = Mock(return_value={})
        with patch("truss.cli.cli.resolve_model_team_name", return_value=(None, None)):
            result = runner.invoke(
                truss_cli,
                [
                    "push",
                    str(custom_model_truss_dir_with_pre_and_post),
                    "--remote",
                    "baseten",
                    "--model-name",
                    "model_name",
                    "--watch",
                ],
            )

    # Should still exit with error on a hard failure
    assert result.exit_code == 1
