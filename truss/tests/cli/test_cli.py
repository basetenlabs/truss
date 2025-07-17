from unittest.mock import Mock, patch

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
                ["push", "test_truss", "--remote", "remote1", "--model-name", "name"],
            )

    assert result.exit_code == 2
    assert (
        "Truss with gRPC transport cannot be used as a development deployment"
        in result.output
    )
