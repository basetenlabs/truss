"""Tests for `truss train capacity update` required-option validation.

The command should report every missing required option at once instead of
surfacing them one at a time across repeated invocations.
"""

from click.testing import CliRunner

from truss.cli.cli import truss_cli


class TestCapacityUpdateMissingArgs:
    def test_no_options_lists_all_missing(self):
        result = CliRunner().invoke(truss_cli, ["train", "capacity", "update"])
        assert result.exit_code != 0
        out = result.output
        assert "--team" in out
        assert "--gpu-type" in out
        assert "--capacity" in out

    def test_partial_options_lists_only_remaining_missing(self):
        result = CliRunner().invoke(
            truss_cli, ["train", "capacity", "update", "--team", "some-team"]
        )
        assert result.exit_code != 0
        out = result.output
        # --team was supplied, so it should not be reported as missing.
        assert "Missing required option(s): --gpu-type, --capacity." in out
