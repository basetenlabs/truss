"""Tests for truss chains CLI commands."""

from unittest.mock import Mock, patch

from click.testing import CliRunner

from truss.cli.cli import truss_cli


def test_chains_push_with_disable_chain_download_flag():
    """Test that --disable-chain-download flag is properly parsed and passed through."""
    runner = CliRunner()

    mock_entrypoint_cls = Mock()
    mock_entrypoint_cls.meta_data.chain_name = "test_chain"
    mock_entrypoint_cls.display_name = "TestChain"

    mock_service = Mock()
    mock_service.run_remote_url = "http://test.com/run_remote"
    mock_service.is_websocket = False

    with patch(
        "truss_chains.framework.ChainletImporter.import_target"
    ) as mock_importer:
        with patch("truss_chains.deployment.deployment_client.push") as mock_push:
            mock_importer.return_value.__enter__.return_value = mock_entrypoint_cls
            mock_push.return_value = mock_service

            result = runner.invoke(
                truss_cli,
                [
                    "chains",
                    "push",
                    "test_chain.py",
                    "--disable-chain-download",
                    "--remote",
                    "test_remote",
                    "--dryrun",
                ],
            )

    assert result.exit_code == 0

    mock_push.assert_called_once()
    call_args = mock_push.call_args
    options = call_args[0][1]

    assert hasattr(options, "disable_chain_download")
    assert options.disable_chain_download is True


def test_chains_push_without_disable_chain_download_flag():
    """Test that disable_chain_download defaults to False when flag is not provided."""
    runner = CliRunner()

    mock_entrypoint_cls = Mock()
    mock_entrypoint_cls.meta_data.chain_name = "test_chain"
    mock_entrypoint_cls.display_name = "TestChain"

    mock_service = Mock()
    mock_service.run_remote_url = "http://test.com/run_remote"
    mock_service.is_websocket = False

    with patch(
        "truss_chains.framework.ChainletImporter.import_target"
    ) as mock_importer:
        with patch("truss_chains.deployment.deployment_client.push") as mock_push:
            mock_importer.return_value.__enter__.return_value = mock_entrypoint_cls
            mock_push.return_value = mock_service

            result = runner.invoke(
                truss_cli,
                [
                    "chains",
                    "push",
                    "test_chain.py",
                    "--remote",
                    "test_remote",
                    "--dryrun",
                ],
            )

    assert result.exit_code == 0

    mock_push.assert_called_once()
    call_args = mock_push.call_args
    options = call_args[0][1]

    assert hasattr(options, "disable_chain_download")
    assert options.disable_chain_download is False


def test_chains_push_help_includes_disable_chain_download():
    """Test that --disable-chain-download appears in the help output."""
    runner = CliRunner()

    result = runner.invoke(truss_cli, ["chains", "push", "--help"])

    assert result.exit_code == 0
    assert "--disable-chain-download" in result.output


def test_chains_push_with_deployment_name_flag():
    """Test that --deployment-name flag is properly parsed and passed through."""
    runner = CliRunner()

    mock_entrypoint_cls = Mock()
    mock_entrypoint_cls.meta_data.chain_name = "test_chain"
    mock_entrypoint_cls.display_name = "TestChain"

    mock_service = Mock()
    mock_service.run_remote_url = "http://test.com/run_remote"
    mock_service.is_websocket = False

    with patch(
        "truss_chains.framework.ChainletImporter.import_target"
    ) as mock_importer:
        with patch("truss_chains.deployment.deployment_client.push") as mock_push:
            mock_importer.return_value.__enter__.return_value = mock_entrypoint_cls
            mock_push.return_value = mock_service

            result = runner.invoke(
                truss_cli,
                [
                    "chains",
                    "push",
                    "test_chain.py",
                    "--deployment-name",
                    "custom_deployment",
                    "--remote",
                    "test_remote",
                    "--publish",
                    "--dryrun",
                ],
            )

    assert result.exit_code == 0

    mock_push.assert_called_once()
    call_args = mock_push.call_args
    options = call_args[0][1]

    assert hasattr(options, "deployment_name")
    assert options.deployment_name == "custom_deployment"
