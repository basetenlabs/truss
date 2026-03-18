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
