"""Tests for truss chains CLI commands."""

from unittest.mock import Mock, patch
from pathlib import Path

import pytest
from click.testing import CliRunner

from truss.cli.cli import truss_cli


def test_chains_push_with_disable_truss_download_flag():
    """Test that --disable-truss-download flag is properly parsed and passed through."""
    runner = CliRunner()
    
    # Mock the chainlet importer and deployment client
    mock_entrypoint_cls = Mock()
    mock_entrypoint_cls.meta_data.chain_name = "test_chain"
    mock_entrypoint_cls.display_name = "TestChain"
    
    mock_service = Mock()
    mock_service.run_remote_url = "http://test.com/run_remote"
    mock_service.is_websocket = False
    
    with patch("truss_chains.framework.ChainletImporter.import_target") as mock_importer:
        with patch("truss_chains.deployment.deployment_client.push") as mock_push:
            mock_importer.return_value.__enter__.return_value = mock_entrypoint_cls
            mock_push.return_value = mock_service
            
            # Test with --disable-truss-download flag
            result = runner.invoke(
                truss_cli,
                [
                    "chains", "push", 
                    "test_chain.py", 
                    "--disable-truss-download",
                    "--remote", "test_remote",
                    "--dryrun"
                ]
            )
    
    assert result.exit_code == 0
    
    # Verify that the deployment client was called with the correct options
    mock_push.assert_called_once()
    call_args = mock_push.call_args
    options = call_args[0][1]  # Second argument is the options
    
    # Check that disable_truss_download is set to True in the options
    assert hasattr(options, 'disable_truss_download')
    assert options.disable_truss_download is True


def test_chains_push_without_disable_truss_download_flag():
    """Test that disable_truss_download defaults to False when flag is not provided."""
    runner = CliRunner()
    
    # Mock the chainlet importer and deployment client
    mock_entrypoint_cls = Mock()
    mock_entrypoint_cls.meta_data.chain_name = "test_chain"
    mock_entrypoint_cls.display_name = "TestChain"
    
    mock_service = Mock()
    mock_service.run_remote_url = "http://test.com/run_remote"
    mock_service.is_websocket = False
    
    with patch("truss_chains.framework.ChainletImporter.import_target") as mock_importer:
        with patch("truss_chains.deployment.deployment_client.push") as mock_push:
            mock_importer.return_value.__enter__.return_value = mock_entrypoint_cls
            mock_push.return_value = mock_service
            
            # Test without --disable-truss-download flag
            result = runner.invoke(
                truss_cli,
                [
                    "chains", "push", 
                    "test_chain.py", 
                    "--remote", "test_remote",
                    "--dryrun"
                ]
            )
    
    assert result.exit_code == 0
    
    # Verify that the deployment client was called with the correct options
    mock_push.assert_called_once()
    call_args = mock_push.call_args
    options = call_args[0][1]  # Second argument is the options
    
    # Check that disable_truss_download defaults to False
    assert hasattr(options, 'disable_truss_download')
    assert options.disable_truss_download is False


def test_chains_push_help_includes_disable_truss_download():
    """Test that --disable-truss-download appears in the help output."""
    runner = CliRunner()
    
    result = runner.invoke(truss_cli, ["chains", "push", "--help"])
    
    assert result.exit_code == 0
    assert "--disable-truss-download" in result.output
    # Check for the help text (it may be wrapped across lines)
    assert "Disable downloading" in result.output
    assert "truss directory" in result.output
