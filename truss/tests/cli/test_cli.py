import threading
from unittest.mock import MagicMock, Mock, patch

import requests
from click.testing import CliRunner

from truss.cli.cli import _keepalive_loop, truss_cli
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
                ["push", "test_truss", "--remote", "remote1", "--model-name", "name"],
            )

    assert result.exit_code == 2
    assert (
        "Truss with gRPC transport cannot be used as a development deployment"
        in result.output
    )


# _keepalive_loop tests


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
                _keepalive_loop(
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
                    _keepalive_loop(
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
                    _keepalive_loop(
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
                    _keepalive_loop(
                        "http://fake/development/sync/v1/models/model",
                        "test_api_key",
                        stop_event,
                    )

    mock_exit.assert_not_called()


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


def test_watch_no_sleep_sends_wake_request():
    """--no-sleep should POST to /development/wake before waiting."""
    resolved_model, versions, dev_version, mock_tr, remote_provider = (
        _make_watch_mocks()
    )
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
                    _result = runner.invoke(
                        truss_cli,
                        ["watch", "/tmp/fake", "--remote", "baseten", "--no-sleep"],
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
        with patch("truss.cli.cli.requests_lib") as mock_requests:
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
    assert kwargs["target"] == _keepalive_loop
    thread_args = kwargs["args"]
    assert (
        thread_args[0]
        == "https://model-abc123.api.baseten.co/development/sync/v1/models/model"
    )
    assert thread_args[1] == "test_key"
    mock_thread.start.assert_called_once()


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
                    "--publish",
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
                    "--publish",
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
                    "--publish",
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
