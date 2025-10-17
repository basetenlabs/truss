from unittest.mock import Mock, patch

from click.testing import CliRunner

from truss.cli.cli import truss_cli


def test_push_with_grpc_transport_fails_for_development_deployment():
    mock_truss = Mock()
    mock_truss.spec.config.runtime.transport.kind = "grpc"

    runner = CliRunner()

    # Test that gRPC transport fails with --watch (development deployment)
    with patch("truss.cli.cli._get_truss_from_directory", return_value=mock_truss):
        with patch("truss.cli.remote_cli.inquire_remote_name", return_value="remote1"):
            result = runner.invoke(
                truss_cli,
                ["push", "test_truss", "--remote", "remote1", "--model-name", "name", "--watch"],
            )

    assert result.exit_code == 2
    assert (
        "Truss with gRPC transport cannot be used as a development deployment"
        in result.output
    )


def test_push_with_grpc_transport_succeeds_by_default():
    """Test that gRPC transport succeeds by default (published deployment)"""
    mock_truss = Mock()
    mock_truss.spec.config.runtime.transport.kind = "grpc"
    mock_remote_provider = Mock()
    mock_service = Mock()
    mock_service.is_draft = False
    mock_remote_provider.push.return_value = mock_service

    runner = CliRunner()

    with patch("truss.cli.cli._get_truss_from_directory", return_value=mock_truss):
        with patch("truss.cli.remote_cli.inquire_remote_name", return_value="remote1"):
            with patch("truss.cli.cli.RemoteFactory.create", return_value=mock_remote_provider):
                result = runner.invoke(
                    truss_cli,
                    ["push", "test_truss", "--remote", "remote1", "--model-name", "name"],
                )

    # Should succeed now since default is published deployment
    assert result.exit_code == 0


def test_push_watch_and_publish_flags_conflict():
    """Test that --watch and --publish flags cannot be used together"""
    mock_truss = Mock()
    mock_truss.spec.config.runtime.transport.kind = "http"

    runner = CliRunner()

    with patch("truss.cli.cli._get_truss_from_directory", return_value=mock_truss):
        with patch("truss.cli.remote_cli.inquire_remote_name", return_value="remote1"):
            result = runner.invoke(
                truss_cli,
                ["push", "test_truss", "--remote", "remote1", "--model-name", "name", "--watch", "--publish"],
            )

    assert result.exit_code == 2
    assert "Cannot use both --watch and --publish flags" in result.output


def test_push_watch_and_promote_flags_conflict():
    """Test that --watch and --promote flags cannot be used together"""
    mock_truss = Mock()
    mock_truss.spec.config.runtime.transport.kind = "http"

    runner = CliRunner()

    with patch("truss.cli.cli._get_truss_from_directory", return_value=mock_truss):
        with patch("truss.cli.remote_cli.inquire_remote_name", return_value="remote1"):
            result = runner.invoke(
                truss_cli,
                ["push", "test_truss", "--remote", "remote1", "--model-name", "name", "--watch", "--promote"],
            )

    assert result.exit_code == 2
    assert "Cannot use both --watch and --promote flags" in result.output


def test_push_default_behavior_is_published():
    """Test that default push behavior creates published deployment"""
    mock_truss = Mock()
    mock_truss.spec.config.runtime.transport.kind = "http"
    mock_remote_provider = Mock()
    mock_service = Mock()
    mock_service.is_draft = False
    mock_remote_provider.push.return_value = mock_service

    runner = CliRunner()

    with patch("truss.cli.cli._get_truss_from_directory", return_value=mock_truss):
        with patch("truss.cli.remote_cli.inquire_remote_name", return_value="remote1"):
            with patch("truss.cli.cli.RemoteFactory.create", return_value=mock_remote_provider):
                result = runner.invoke(
                    truss_cli,
                    ["push", "test_truss", "--remote", "remote1", "--model-name", "name"],
                )

    # Check that push was called with publish=True
    assert result.exit_code == 0
    mock_remote_provider.push.assert_called_once()
    call_args = mock_remote_provider.push.call_args
    assert call_args.kwargs['publish'] is True
    assert call_args.kwargs['watch'] is False


def test_push_watch_enables_log_streaming():
    """Test that --watch flag enables log streaming and watch functionality"""
    mock_truss = Mock()
    mock_truss.spec.config.runtime.transport.kind = "http"
    mock_remote_provider = Mock()
    mock_service = Mock()
    mock_service.is_draft = True  # Development deployment
    mock_service.model_id = "test_model_id"
    mock_service.model_version_id = "test_version_id"
    mock_remote_provider.push.return_value = mock_service

    runner = CliRunner()

    with patch("truss.cli.cli._get_truss_from_directory", return_value=mock_truss):
        with patch("truss.cli.remote_cli.inquire_remote_name", return_value="remote1"):
            with patch("truss.cli.cli.RemoteFactory.create", return_value=mock_remote_provider):
                # Mock the log watcher to avoid actual log streaming
                with patch("truss.cli.cli.ModelDeploymentLogWatcher") as mock_log_watcher:
                    mock_log_watcher.return_value.watch.return_value = []
                    
                    # Mock the sync function to avoid actual file watching
                    with patch.object(mock_remote_provider, 'sync_truss_to_dev_version_by_name'):
                        result = runner.invoke(
                            truss_cli,
                            ["push", "test_truss", "--remote", "remote1", "--model-name", "name", "--watch"],
                        )

    # Should succeed and call with watch=True
    assert result.exit_code == 0
    call_args = mock_remote_provider.push.call_args
    assert call_args.kwargs['publish'] is False  # Development deployment
    assert call_args.kwargs['watch'] is True
    
    # Verify log watcher was called
    mock_log_watcher.assert_called_once_with(
        mock_remote_provider.api, "test_model_id", "test_version_id"
    )
