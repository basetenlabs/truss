import os
from unittest.mock import patch

import click
import pytest

from truss.base import truss_config
from truss.cli.utils import common


class TestCheckIsInteractive:
    @patch("truss.cli.utils.common.sys.stdin")
    @patch("truss.cli.utils.common.sys.stdout")
    def test_non_interactive_when_stdin_not_tty(self, mock_stdout, mock_stdin):
        mock_stdin.isatty.return_value = False
        mock_stdout.isatty.return_value = True
        assert common.check_is_interactive() is False

    @patch("truss.cli.utils.common.sys.stdin")
    @patch("truss.cli.utils.common.sys.stdout")
    def test_non_interactive_when_stdout_not_tty(self, mock_stdout, mock_stdin):
        mock_stdin.isatty.return_value = True
        mock_stdout.isatty.return_value = False
        assert common.check_is_interactive() is False

    @patch("truss.cli.utils.common.sys.stdin")
    @patch("truss.cli.utils.common.sys.stdout")
    def test_non_interactive_flag_overrides_tty(self, mock_stdout, mock_stdin):
        mock_stdin.isatty.return_value = True
        mock_stdout.isatty.return_value = True

        ctx = click.Context(click.Command("test"), obj={"non_interactive": True})
        with ctx:
            assert common.check_is_interactive() is False

    @patch("truss.cli.utils.common.sys.stdin")
    @patch("truss.cli.utils.common.sys.stdout")
    def test_interactive_when_tty_and_no_flag(self, mock_stdout, mock_stdin):
        mock_stdin.isatty.return_value = True
        mock_stdout.isatty.return_value = True

        ctx = click.Context(click.Command("test"), obj={"non_interactive": False})
        with ctx:
            assert common.check_is_interactive() is True

    @patch("truss.cli.utils.common.sys.stdin")
    @patch("truss.cli.utils.common.sys.stdout")
    def test_interactive_when_context_obj_is_none(self, mock_stdout, mock_stdin):
        mock_stdin.isatty.return_value = True
        mock_stdout.isatty.return_value = True

        ctx = click.Context(click.Command("test"))
        with ctx:
            assert common.check_is_interactive() is True


@patch("truss.cli.utils.common.check_is_interactive", return_value=True)
@patch("truss.cli.utils.common.self_upgrade.notify_baseten_cli")
@patch("truss.cli.utils.common.self_upgrade.notify_if_outdated")
class TestUpgradeDialogue:
    def test_skips_check_when_opted_out(
        self, mock_notify, mock_notify_cli, mock_interactive
    ):
        for value in ("1", "true", "TRUE"):
            with patch.dict(os.environ, {"TRUSS_NO_UPDATE_CHECK": value}):
                common.maybe_upgrade_dialogue()
        mock_notify.assert_not_called()
        mock_notify_cli.assert_not_called()

    def test_runs_check_when_not_opted_out(
        self, mock_notify, mock_notify_cli, mock_interactive
    ):
        for value in ("", "0", "false"):
            with patch.dict(os.environ, {"TRUSS_NO_UPDATE_CHECK": value}):
                common.maybe_upgrade_dialogue()
        assert mock_notify.call_count == 3
        assert mock_notify_cli.call_count == 3

    def test_skips_baseten_cli_banner_when_disabled(
        self, mock_notify, mock_notify_cli, mock_interactive
    ):
        with patch.dict(os.environ, {"TRUSS_NO_UPDATE_CHECK": ""}):
            common.maybe_upgrade_dialogue(baseten_cli_banner=False)
        mock_notify.assert_called_once()
        mock_notify_cli.assert_not_called()

    def test_skips_baseten_cli_banner_when_not_interactive(
        self, mock_notify, mock_notify_cli, mock_interactive
    ):
        mock_interactive.return_value = False
        with patch.dict(os.environ, {"TRUSS_NO_UPDATE_CHECK": ""}):
            common.maybe_upgrade_dialogue()
        mock_notify.assert_called_once()
        mock_notify_cli.assert_not_called()

    def test_baseten_cli_banner_runs_when_upgrade_check_fails(
        self, mock_notify, mock_notify_cli, mock_interactive
    ):
        mock_notify.side_effect = Exception("Network error")
        with patch.dict(os.environ, {"TRUSS_NO_UPDATE_CHECK": ""}):
            common.maybe_upgrade_dialogue()
        mock_notify_cli.assert_called_once()


def test_keepalive_ping_path_truss_server_default():
    config = truss_config.TrussConfig()
    assert common.keepalive_ping_path(config) == "v1/models/model"


def test_keepalive_ping_path_docker_server_uses_readiness_endpoint():
    config = truss_config.TrussConfig(
        docker_server=truss_config.DockerServer(
            start_command="python server.py",
            server_port=8000,
            predict_endpoint="/v1/chat/completions",
            readiness_endpoint="/ready",
            liveness_endpoint="/health",
        )
    )
    assert common.keepalive_ping_path(config) == "ready"


def test_keepalive_url_draft():
    url = common._keepalive_url("https://model-abc.api.baseten.co", True, None, "ready")
    assert url == "https://model-abc.api.baseten.co/development/sync/ready"


def test_keepalive_url_published():
    url = common._keepalive_url(
        "https://model-abc.api.baseten.co", False, "dep-id", "v1/models/model"
    )
    assert url == (
        "https://model-abc.api.baseten.co/deployment/dep-id/sync/v1/models/model"
    )


def test_keepalive_url_published_requires_deployment_id():
    with pytest.raises(ValueError):
        common._keepalive_url(
            "https://model-abc.api.baseten.co", False, None, "v1/models/model"
        )


def test_normalize_iso_timestamp_handles_nanoseconds():
    normalized = common._normalize_iso_timestamp("2025-11-17 05:05:06.000000000 +0000")
    assert normalized == "2025-11-17 05:05:06.000000+00:00"


def test_normalize_iso_timestamp_handles_z_suffix_and_short_fraction():
    normalized = common._normalize_iso_timestamp("2025-11-17T05:05:06.123456Z")
    assert normalized == "2025-11-17T05:05:06.123456+00:00"
