from unittest.mock import MagicMock, Mock, patch

from click.testing import CliRunner

from truss.cli.cli import truss_cli
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
    assert "OIDC Information" in result.output
    assert "PJAd5Q0" in result.output
    assert "wgeyxoq (Default Team)" in result.output
    assert "abc123 (ML Team)" in result.output
    assert "https://oidc.baseten.co" in result.output
    assert "oidc.baseten.co" in result.output
    assert "Example Subject Claims" in result.output
    normalized_output = result.output.replace("\n", "")
    assert (
        "v=1:org=PJAd5Q0:team=wgeyxoq:"
        "model=<model_id>:deployment=<deployment_id>:env=<environment>:type=model_build"
        in normalized_output
    )
    assert (
        "v=1:org=PJAd5Q0:team=wgeyxoq:"
        "model=<model_id>:deployment=<deployment_id>:env=<environment>:type=model_container"
        in normalized_output