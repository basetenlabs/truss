"""Comprehensive end-to-end tests for chain upload flow using Click test runner."""

from unittest.mock import Mock, patch

import pytest
from click.testing import CliRunner

from truss.cli.cli import truss_cli
from truss.remote.baseten.core import ChainDeploymentHandleAtomic

CHAINLET_CONTENT = """
import truss_chains as chains

@chains.mark_entrypoint
class TestChainlet(chains.ChainletBase):
def run_remote( input_data):
    return {"result": "test_output"}
"""


@pytest.fixture
def mock_chainlet_file(tmp_path):
    """Create a temporary chainlet file for testing."""
    chainlet_file = tmp_path / "test_chainlet.py"
    chainlet_file.write_text(CHAINLET_CONTENT)
    return chainlet_file


@pytest.fixture
def mock_entrypoint_cls(mock_chainlet_file):
    """Create a mock entrypoint chainlet class."""
    mock_entrypoint_cls = Mock()
    mock_entrypoint_cls.meta_data.chain_name = "test_chain"
    mock_entrypoint_cls.display_name = "TestChainlet"
    mock_entrypoint_cls.__file__ = str(mock_chainlet_file)
    return mock_entrypoint_cls


@pytest.fixture
def mock_service():
    """Create a mock BasetenChainService."""
    mock_service = Mock()
    mock_service.run_remote_url = "http://test.com/run_remote"
    mock_service.is_websocket = False
    mock_service.name = "test_chain"
    return mock_service


@pytest.fixture
def mock_deployment_handle():
    """Create a mock ChainDeploymentHandleAtomic."""
    return ChainDeploymentHandleAtomic(
        chain_deployment_id="test_deployment_id",
        chain_id="test_chain_id",
        hostname="test_hostname",
        is_draft=False,
    )


@pytest.fixture
def cli_runner():
    """Create a Click CLI runner."""
    return CliRunner()


@pytest.fixture
def mock_external_apis(mock_entrypoint_cls, mock_service):
    """Mock all external API calls and yield the mocks for verification."""
    with patch(
        "truss_chains.framework.ChainletImporter.import_target"
    ) as mock_importer:
        with patch("truss_chains.deployment.deployment_client.push") as mock_push:
            mock_importer.return_value.__enter__.return_value = mock_entrypoint_cls
            mock_push.return_value = mock_service

            yield {"importer": mock_importer, "push": mock_push}


def test_push_chain_with_disable_chain_download_flag(
    mock_chainlet_file, mock_external_apis, cli_runner
):
    """Test complete chain upload flow with --disable-chain-download flag."""
    result = cli_runner.invoke(
        truss_cli,
        [
            "chains",
            "push",
            str(mock_chainlet_file),
            "--disable-chain-download",
            "--remote",
            "test_remote",
            "--dryrun",
        ],
    )

    assert result.exit_code == 0

    mock_push = mock_external_apis["push"]
    mock_push.assert_called_once()
    call_args = mock_push.call_args
    options = call_args[0][1]

    assert hasattr(options, "disable_chain_download")
    assert options.disable_chain_download is True
    assert options.chain_name == "test_chain"


def test_push_chain_without_disable_chain_download_flag(
    mock_chainlet_file, mock_external_apis, cli_runner
):
    """Test complete chain upload flow without --disable-chain-download flag (defaults to False)."""
    result = cli_runner.invoke(
        truss_cli,
        [
            "chains",
            "push",
            str(mock_chainlet_file),
            "--remote",
            "test_remote",
            "--dryrun",
        ],
    )

    assert result.exit_code == 0

    mock_push = mock_external_apis["push"]
    mock_push.assert_called_once()
    call_args = mock_push.call_args
    options = call_args[0][1]

    assert hasattr(options, "disable_chain_download")
    assert options.disable_chain_download is False
    assert options.chain_name == "test_chain"


def test_push_chain_with_all_parameters_including_disable_chain_download(
    mock_chainlet_file, mock_external_apis, cli_runner
):
    """Test complete chain upload flow with all parameters including disable_chain_download."""
    result = cli_runner.invoke(
        truss_cli,
        [
            "chains",
            "push",
            str(mock_chainlet_file),
            "--name",
            "custom_chain_name",
            "--disable-chain-download",
            "--publish",
            "--no-promote",
            "--environment",
            "test_env",
            "--include-git-info",
            "--remote",
            "test_remote",
            "--dryrun",
        ],
    )

    assert result.exit_code == 0

    mock_push = mock_external_apis["push"]
    mock_push.assert_called_once()
    call_args = mock_push.call_args
    options = call_args[0][1]

    assert options.chain_name == "custom_chain_name"
    assert options.publish is True
    assert options.environment == "test_env"
    assert options.include_git_info is True
    assert options.disable_chain_download is True
