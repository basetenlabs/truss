from unittest.mock import patch

import click

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


def test_normalize_iso_timestamp_handles_nanoseconds():
    normalized = common._normalize_iso_timestamp("2025-11-17 05:05:06.000000000 +0000")
    assert normalized == "2025-11-17 05:05:06.000000+00:00"


def test_normalize_iso_timestamp_handles_z_suffix_and_short_fraction():
    normalized = common._normalize_iso_timestamp("2025-11-17T05:05:06.123456Z")
    assert normalized == "2025-11-17T05:05:06.123456+00:00"
