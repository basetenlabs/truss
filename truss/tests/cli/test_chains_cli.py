"""Tests for truss chains CLI commands."""

import os
from contextlib import contextmanager
from unittest.mock import Mock, patch

import rich.table
from click.testing import CliRunner

from truss.cli.cli import truss_cli
from truss.remote.baseten import custom_types as b10_types
from truss_chains.deployment.deployment_client import BasetenChainService


def _active_chainlet() -> b10_types.DeployedChainlet:
    return b10_types.DeployedChainlet(
        name="Entrypoint",
        is_entrypoint=True,
        is_draft=True,
        status="ACTIVE",
        logs_url="https://app.baseten.co/logs",
        oracle_name="Entrypoint-oracle",
        oracle_id="oracle-id",
        oracle_version_id="version-id",
        hostname="https://model-abc.api.baseten.co",
    )


def _mock_baseten_chain_service() -> BasetenChainService:
    service = object.__new__(BasetenChainService)
    service._name = "test_chain"
    service._entrypoint_descriptor = None
    service._chain_deployment_handle = Mock(
        hostname="chain.api.baseten.co",
        chain_deployment_id="deployment_id",
        is_draft=True,
    )
    # The push wait loop polls `get_info()` to drive both the status table and
    # incremental keepalive; return a single already-ACTIVE chainlet so the loop
    # completes immediately.
    service.get_info = Mock(return_value=[_active_chainlet()])  # type: ignore[method-assign]
    return service


@contextmanager
def _patch_chains_push_watch_flow(mock_watch):
    mock_entrypoint_cls = Mock()
    mock_entrypoint_cls.meta_data.chain_name = "test_chain"
    mock_entrypoint_cls.display_name = "TestChain"

    mock_options = Mock()
    mock_options.environment = None

    mock_remote_provider = Mock()
    mock_remote_provider.api.get_teams.return_value = {}

    with patch(
        "truss_chains.framework.ChainletImporter.import_target"
    ) as mock_importer:
        with patch(
            "truss.cli.chains_commands.RemoteFactory.create",
            return_value=mock_remote_provider,
        ):
            with patch(
                "truss.cli.chains_commands.resolve_chain_team_name",
                return_value=(None, None),
            ):
                with patch(
                    "truss_chains.private_types.PushOptionsBaseten.create",
                    return_value=mock_options,
                ):
                    with patch(
                        "truss_chains.deployment.deployment_client.push"
                    ) as mock_push:
                        with patch(
                            "truss_chains.deployment.deployment_client.watch",
                            mock_watch,
                        ):
                            with patch(
                                "truss.cli.chains_commands._build_chains_table",
                                return_value=(rich.table.Table(), ["ACTIVE"]),
                            ):
                                with patch(
                                    "truss.cli.chains_commands._make_chains_curl_snippet",
                                    return_value="curl http://test.com",
                                ):
                                    with patch(
                                        "truss_chains.deployment.deployment_client.cli_common.start_keepalive"
                                    ):
                                        mock_importer.return_value.__enter__.return_value = mock_entrypoint_cls
                                        mock_push.return_value = (
                                            _mock_baseten_chain_service()
                                        )
                                        yield mock_watch


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


def test_chains_push_help_includes_watch_no_sleep():
    """Test that --watch-no-sleep appears in chains push help output."""
    env = os.environ.copy()
    env["COLUMNS"] = "200"
    runner = CliRunner(env=env)

    result = runner.invoke(truss_cli, ["chains", "push", "--help"])

    assert result.exit_code == 0
    assert "--watch-no-sleep" in result.output


def test_chains_watch_help_includes_no_sleep():
    """Test that --no-sleep appears in chains watch help output."""
    env = os.environ.copy()
    env["COLUMNS"] = "200"
    runner = CliRunner(env=env)

    result = runner.invoke(truss_cli, ["chains", "watch", "--help"])

    assert result.exit_code == 0
    assert "--no-sleep" in result.output


def test_chains_push_watch_no_sleep_without_watch_fails():
    """--watch-no-sleep without --watch should fail."""
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
                "--watch-no-sleep=false",
                "--remote",
                "test_remote",
            ],
        )

    assert result.exit_code != 0
    assert "--watch-no-sleep requires --watch" in result.output or (
        result.exception
        and "--watch-no-sleep requires --watch" in str(result.exception.__context__)
    )


def test_chains_watch_defaults_to_no_sleep():
    """chains watch should default to keepalive enabled, matching truss watch."""
    runner = CliRunner()

    with patch("truss_chains.deployment.deployment_client.watch") as mock_watch:
        result = runner.invoke(
            truss_cli, ["chains", "watch", "test_chain.py", "--remote", "test_remote"]
        )

    assert result.exit_code == 0
    mock_watch.assert_called_once()
    assert mock_watch.call_args.kwargs["no_sleep"] is True


def test_chains_watch_without_no_sleep_disables_keepalive():
    """chains watch --no-sleep=false should disable keepalive."""
    runner = CliRunner()

    with patch("truss_chains.deployment.deployment_client.watch") as mock_watch:
        result = runner.invoke(
            truss_cli,
            [
                "chains",
                "watch",
                "test_chain.py",
                "--remote",
                "test_remote",
                "--no-sleep=false",
            ],
        )

    assert result.exit_code == 0
    mock_watch.assert_called_once()
    assert mock_watch.call_args.kwargs["no_sleep"] is False


def test_chains_push_watch_starts_keepalive_during_wait_loop():
    """During `push --watch`, ready chainlets are warmed before watch starts, and
    the warmed set is shared with watch to avoid duplicate keepalive threads."""
    runner = CliRunner()
    mock_watch = Mock()

    with _patch_chains_push_watch_flow(mock_watch):
        with patch(
            "truss_chains.deployment.deployment_client._start_keepalives_for_ready_chainlets"
        ) as mock_start:
            result = runner.invoke(
                truss_cli,
                [
                    "chains",
                    "push",
                    "test_chain.py",
                    "--watch",
                    "--remote",
                    "test_remote",
                ],
            )

    assert result.exit_code == 0
    # Keepalive was started incrementally during the wait loop.
    assert mock_start.call_count >= 1
    # The same warmed-set object is forwarded to watch.
    push_phase_set = mock_start.call_args.args[2]
    assert mock_watch.call_args.kwargs["started_keepalives"] is push_phase_set


def test_chains_push_watch_no_sleep_false_still_warms_during_wait_loop():
    """A push always keeps ready chainlets warm during the wait loop, even with
    `--watch-no-sleep=false`. That flag only governs the subsequent watch phase."""
    runner = CliRunner()
    mock_watch = Mock()

    with _patch_chains_push_watch_flow(mock_watch):
        with patch(
            "truss_chains.deployment.deployment_client._start_keepalives_for_ready_chainlets"
        ) as mock_start:
            result = runner.invoke(
                truss_cli,
                [
                    "chains",
                    "push",
                    "test_chain.py",
                    "--watch",
                    "--watch-no-sleep=false",
                    "--remote",
                    "test_remote",
                ],
            )

    assert result.exit_code == 0
    # Wait-loop keepalive still runs for the development push.
    assert mock_start.call_count >= 1
    # But `watch_no_sleep=false` is forwarded so the watch phase allows sleeping.
    assert mock_watch.call_args.kwargs["no_sleep"] is False


def test_chains_push_watch_defaults_watch_no_sleep():
    """chains push --watch should enable keepalive by default."""
    runner = CliRunner()
    mock_watch = Mock()

    with _patch_chains_push_watch_flow(mock_watch):
        result = runner.invoke(
            truss_cli,
            ["chains", "push", "test_chain.py", "--watch", "--remote", "test_remote"],
        )

    assert result.exit_code == 0
    mock_watch.assert_called_once()
    assert mock_watch.call_args.kwargs["no_sleep"] is True


def test_chains_push_watch_watch_no_sleep_false_disables_keepalive():
    """chains push --watch --watch-no-sleep=false should disable keepalive."""
    runner = CliRunner()
    mock_watch = Mock()

    with _patch_chains_push_watch_flow(mock_watch):
        result = runner.invoke(
            truss_cli,
            [
                "chains",
                "push",
                "test_chain.py",
                "--watch",
                "--watch-no-sleep=false",
                "--remote",
                "test_remote",
            ],
        )

    assert result.exit_code == 0
    mock_watch.assert_called_once()
    assert mock_watch.call_args.kwargs["no_sleep"] is False
