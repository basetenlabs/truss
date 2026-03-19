from unittest.mock import patch

import click
import pytest

from truss.cli.remote_cli import inquire_remote_name


class TestInquireRemoteName:
    @patch(
        "truss.cli.remote_cli.RemoteFactory.get_available_config_names",
        return_value=["remote1", "remote2"],
    )
    @patch("truss.cli.remote_cli.check_is_interactive", return_value=False)
    def test_multiple_remotes_non_interactive_raises(
        self, _mock_interactive, _mock_remotes
    ):
        with pytest.raises(click.UsageError, match="--remote"):
            inquire_remote_name()

    @patch(
        "truss.cli.remote_cli.RemoteFactory.get_available_config_names", return_value=[]
    )
    @patch("truss.cli.remote_cli.check_is_interactive", return_value=False)
    def test_no_remotes_non_interactive_raises(self, _mock_interactive, _mock_remotes):
        with pytest.raises(click.UsageError, match="--remote"):
            inquire_remote_name()

    @patch(
        "truss.cli.remote_cli.RemoteFactory.get_available_config_names",
        return_value=["only-one"],
    )
    def test_single_remote_returns_without_prompt(self, _mock_remotes):
        assert inquire_remote_name() == "only-one"


class TestCheckIsInteractive:
    @patch("truss.cli.utils.common.sys.stdin")
    @patch("truss.cli.utils.common.sys.stdout")
    def test_non_interactive_when_stdin_not_tty(self, mock_stdout, mock_stdin):
        from truss.cli.utils.common import check_is_interactive

        mock_stdin.isatty.return_value = False
        mock_stdout.isatty.return_value = True
        assert check_is_interactive() is False

    @patch("truss.cli.utils.common.sys.stdin")
    @patch("truss.cli.utils.common.sys.stdout")
    def test_non_interactive_when_stdout_not_tty(self, mock_stdout, mock_stdin):
        from truss.cli.utils.common import check_is_interactive

        mock_stdin.isatty.return_value = True
        mock_stdout.isatty.return_value = False
        assert check_is_interactive() is False

    @patch("truss.cli.utils.common.sys.stdin")
    @patch("truss.cli.utils.common.sys.stdout")
    def test_non_interactive_flag_overrides_tty(self, mock_stdout, mock_stdin):
        from truss.cli.utils.common import check_is_interactive

        mock_stdin.isatty.return_value = True
        mock_stdout.isatty.return_value = True

        ctx = click.Context(click.Command("test"), obj={"non_interactive": True})
        with ctx:
            assert check_is_interactive() is False

    @patch("truss.cli.utils.common.sys.stdin")
    @patch("truss.cli.utils.common.sys.stdout")
    def test_interactive_when_tty_and_no_flag(self, mock_stdout, mock_stdin):
        from truss.cli.utils.common import check_is_interactive

        mock_stdin.isatty.return_value = True
        mock_stdout.isatty.return_value = True

        ctx = click.Context(click.Command("test"), obj={"non_interactive": False})
        with ctx:
            assert check_is_interactive() is True

    @patch("truss.cli.utils.common.sys.stdin")
    @patch("truss.cli.utils.common.sys.stdout")
    def test_interactive_when_context_obj_is_none(self, mock_stdout, mock_stdin):
        from truss.cli.utils.common import check_is_interactive

        mock_stdin.isatty.return_value = True
        mock_stdout.isatty.return_value = True

        ctx = click.Context(click.Command("test"))
        with ctx:
            assert check_is_interactive() is True
