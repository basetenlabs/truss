"""Tests for team parameter in training project creation."""

from pathlib import Path
from unittest.mock import Mock, patch

from click.testing import CliRunner

from truss.cli.cli import truss_cli
from truss.remote.baseten.remote import BasetenRemote


class TestTeamParameter:
    """Test team parameter in training project creation."""

    @staticmethod
    def _setup_mock_remote(teams):
        """Setup mock remote with teams."""
        mock_remote = Mock(spec=BasetenRemote)
        mock_api = Mock()
        mock_remote.api = mock_api
        mock_api.get_teams.return_value = teams
        return mock_remote

    @staticmethod
    def _create_test_config():
        """Create a temporary test config file."""
        config_path = Path("/tmp/test_config.py")
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text("# dummy config")
        return config_path

    @staticmethod
    def _invoke_train_push(runner, config_path, team_name=None, remote="test_remote"):
        """Invoke truss train push command."""
        args = ["train", "push", str(config_path), "--remote", remote]
        if team_name:
            args.extend(["--team", team_name])
        return runner.invoke(truss_cli, args)

    @patch("truss_train.deployment.create_training_job_from_file")
    @patch("truss.cli.train_commands.RemoteFactory.create")
    @patch("truss.cli.train_commands.console.status")
    def test_team_not_provided_single_team_uses_default(
        self, mock_status, mock_remote_factory, mock_create_job
    ):
        """Test that default team is used when --team not provided and user has 1 team."""
        mock_status.return_value.__enter__ = Mock(return_value=None)
        mock_status.return_value.__exit__ = Mock(return_value=None)

        teams = {"Team Alpha": {"id": "team1", "name": "Team Alpha"}}
        mock_remote = self._setup_mock_remote(teams)
        mock_remote_factory.return_value = mock_remote

        mock_create_job.return_value = {
            "id": "job123",
            "training_project": {"id": "12345", "name": "test-project"},
        }

        runner = CliRunner()
        config_path = self._create_test_config()

        result = self._invoke_train_push(runner, config_path)

        assert result.exit_code == 0
        mock_create_job.assert_called_once()
        assert mock_create_job.call_args[1]["team_id"] == "team1"

    @patch("truss_train.deployment.create_training_job_from_file")
    @patch("truss.cli.train_commands.RemoteFactory.create")
    @patch("truss.cli.remote_cli.inquire_team")
    @patch("truss.cli.train_commands.console.status")
    def test_team_not_provided_multiple_teams_inquires(
        self, mock_status, mock_inquire_team, mock_remote_factory, mock_create_job
    ):
        """Test that inquire_team is called when --team not provided and user has > 1 team."""
        mock_status.return_value.__enter__ = Mock(return_value=None)
        mock_status.return_value.__exit__ = Mock(return_value=None)

        teams = {
            "Team Alpha": {"id": "team1", "name": "Team Alpha"},
            "Team Beta": {"id": "team2", "name": "Team Beta"},
        }
        mock_remote = self._setup_mock_remote(teams)
        mock_remote_factory.return_value = mock_remote

        mock_inquire_team.return_value = "team2"
        mock_create_job.return_value = {
            "id": "job123",
            "training_project": {"id": "12345", "name": "test-project"},
        }

        runner = CliRunner()
        config_path = self._create_test_config()

        result = self._invoke_train_push(runner, config_path)

        assert result.exit_code == 0
        mock_inquire_team.assert_called_once()
        assert mock_inquire_team.call_args[1]["existing_teams"] == teams
        mock_create_job.assert_called_once()
        assert mock_create_job.call_args[1]["team_id"] == "team2"

    @patch("truss_train.deployment.create_training_job_from_file")
    @patch("truss.cli.train_commands.RemoteFactory.create")
    @patch("truss.cli.train_commands.console.status")
    def test_team_provided_valid_propagated_to_backend(
        self, mock_status, mock_remote_factory, mock_create_job
    ):
        """Test that --team parameter is propagated to backend request when valid."""
        mock_status.return_value.__enter__ = Mock(return_value=None)
        mock_status.return_value.__exit__ = Mock(return_value=None)

        teams = {"Team Alpha": {"id": "team1", "name": "Team Alpha"}}
        mock_remote = self._setup_mock_remote(teams)
        mock_remote_factory.return_value = mock_remote

        mock_create_job.return_value = {
            "id": "job123",
            "training_project": {"id": "12345", "name": "test-project"},
        }

        runner = CliRunner()
        config_path = self._create_test_config()

        result = self._invoke_train_push(runner, config_path, team_name="Team Alpha")

        assert result.exit_code == 0
        mock_create_job.assert_called_once()
        assert mock_create_job.call_args[1]["team_id"] == "team1"

    @patch("truss.cli.train_commands.RemoteFactory.create")
    def test_team_provided_invalid_raises_exception(self, mock_remote_factory):
        """Test that invalid --team raises exception and command fails."""
        teams = {"Team Alpha": {"id": "team1", "name": "Team Alpha"}}
        mock_remote = self._setup_mock_remote(teams)
        mock_remote_factory.return_value = mock_remote

        runner = CliRunner()
        config_path = self._create_test_config()

        result = self._invoke_train_push(runner, config_path, team_name="Invalid Team")

        assert result.exit_code != 0
        assert "does not exist" in result.output
        assert "Invalid Team" in result.output
        assert "Team" in result.output and "Alpha" in result.output
