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
