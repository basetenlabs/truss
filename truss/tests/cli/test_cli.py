from unittest.mock import MagicMock, Mock, patch

from click.testing import CliRunner

from truss.cli.cli import truss_cli


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
        result.exception and "Cannot use --watch with --promote" in str(result.exception.__context__)
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
    assert "Cannot use --watch with --environment" in result.output


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
    assert "Cannot use --watch with --tail" in result.output
