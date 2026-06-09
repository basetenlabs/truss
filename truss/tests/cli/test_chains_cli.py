"""Tests for truss chains CLI commands."""

import os
from unittest.mock import Mock, patch

from click.testing import CliRunner

from truss.cli.cli import truss_cli
from truss.remote.baseten.core import CHAINLET_READY_STATUSES, DEPLOYING_STATUSES


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
    # Use a wide terminal to prevent rich-click from truncating option names.
    env = os.environ.copy()
    env["COLUMNS"] = "200"
    runner = CliRunner(env=env)

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
                    "--dryrun",
                ],
            )

    assert result.exit_code == 0

    mock_push.assert_called_once()
    call_args = mock_push.call_args
    options = call_args[0][1]

    assert hasattr(options, "deployment_name")
    assert options.deployment_name == "custom_deployment"


def test_chains_push_defaults_to_published():
    """Test that chains push defaults to published deployment."""
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

    # Verify publish defaults to True (published deployment)
    assert options.publish is True


def test_chains_push_publish_flag_shows_deprecation_warning():
    """Test that --publish flag shows deprecation warning."""
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
                    "--publish",
                    "--remote",
                    "test_remote",
                    "--dryrun",
                ],
            )

    assert result.exit_code == 0
    assert "DEPRECATED" in result.output


def test_chains_push_no_publish_flag_shows_deprecation_warning():
    """Test that --no-publish flag shows deprecation warning but still works."""
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
                    "--no-publish",
                    "--remote",
                    "test_remote",
                    "--dryrun",
                ],
            )

    assert result.exit_code == 0
    assert "DEPRECATED" in result.output

    # Verify it still creates development deployment for backwards compatibility
    mock_push.assert_called_once()
    call_args = mock_push.call_args
    options = call_args[0][1]
    assert options.publish is False


def test_chains_push_watch_creates_development_deployment():
    """Test that --watch creates a development deployment."""
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
                    "--watch",
                    "--remote",
                    "test_remote",
                    "--dryrun",
                ],
            )

    assert result.exit_code == 0

    mock_push.assert_called_once()
    call_args = mock_push.call_args
    options = call_args[0][1]

    # Verify --watch creates development deployment
    assert options.publish is False


def test_chains_push_watch_with_promote_fails():
    """Test that --watch with --promote fails."""
    runner = CliRunner()

    mock_entrypoint_cls = Mock()
    mock_entrypoint_cls.meta_data.chain_name = "test_chain"
    mock_entrypoint_cls.display_name = "TestChain"

    with patch(
        "truss_chains.framework.ChainletImporter.import_target"
    ) as mock_importer:
        mock_importer.return_value.__enter__.return_value = mock_entrypoint_cls

        result = runner.invoke(
            truss_cli,
            [
                "chains",
                "push",
                "test_chain.py",
                "--watch",
                "--promote",
                "--remote",
                "test_remote",
            ],
        )

    assert result.exit_code == 1
    assert "the deployment cannot be published" in result.output


def test_chains_push_watch_with_environment_fails():
    """Test that --watch with --environment fails."""
    runner = CliRunner()

    mock_entrypoint_cls = Mock()
    mock_entrypoint_cls.meta_data.chain_name = "test_chain"
    mock_entrypoint_cls.display_name = "TestChain"

    with patch(
        "truss_chains.framework.ChainletImporter.import_target"
    ) as mock_importer:
        mock_importer.return_value.__enter__.return_value = mock_entrypoint_cls

        result = runner.invoke(
            truss_cli,
            [
                "chains",
                "push",
                "test_chain.py",
                "--watch",
                "--environment",
                "staging",
                "--remote",
                "test_remote",
            ],
        )

    assert result.exit_code == 1
    assert "Cannot use --watch with --environment" in result.output


def _make_chainlet_info(name: str, status: str, is_entrypoint: bool = False):
    chainlet_info = Mock()
    chainlet_info.is_entrypoint = is_entrypoint
    chainlet_info.name = name
    chainlet_info.status = status
    chainlet_info.logs_url = f"https://app.baseten.co/chains/test/logs/{name}"
    return chainlet_info


def test_create_chains_table_treats_scaled_to_zero_as_ready():
    """Scaled-to-zero chainlets are ready, not failed."""
    from truss.cli.chains_commands import _create_chains_table

    mock_service = Mock()
    mock_service.name = "TestChain"
    mock_service.status_page_url = "https://app.baseten.co/chains/test123/overview"
    mock_service.get_info.return_value = [
        _make_chainlet_info("TestChain", "MODEL_READY", is_entrypoint=True),
        _make_chainlet_info("Worker", "SCALED_TO_ZERO"),
    ]

    _, statuses = _create_chains_table(mock_service)

    assert statuses == ["ACTIVE", "SCALED_TO_ZERO"]


def test_chainlet_ready_statuses_include_scaled_to_zero():
    statuses = ["ACTIVE", "SCALED_TO_ZERO", "BUILDING"]
    num_ready = sum(s in CHAINLET_READY_STATUSES for s in statuses)
    num_deploying = sum(s in DEPLOYING_STATUSES for s in statuses)
    assert num_ready == 2
    assert num_deploying == 1
    assert len(statuses) - num_ready - num_deploying == 0
