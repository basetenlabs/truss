"""Tests for team parameter in training project creation."""

from pathlib import Path
from unittest.mock import Mock, patch

from click.testing import CliRunner
from requests import Response

from truss.cli.cli import truss_cli
from truss.remote.baseten.remote import BasetenRemote


def mock_upsert_training_project_response():
    """Create a mock response for upsert_training_project."""
    response = Response()
    response.status_code = 200
    response.json = Mock(
        return_value={"training_project": {"id": "12345", "name": "training-project"}}
    )
    return response


class TestTeamParameter:
    """Test team parameter in training project creation."""

    @patch("truss_train.deployment.create_training_job_from_file")
    @patch("truss.cli.train_commands.RemoteFactory.create")
    @patch("truss.cli.train_commands.console.status")
    @patch("truss.cli.train_commands._handle_post_create_logic")
    def test_team_provided_propagated_to_backend(
        self, mock_post_create, mock_status, mock_remote_factory, mock_create_job
    ):
        """Test that --team parameter is propagated to backend request."""
        mock_status.return_value.__enter__ = Mock(return_value=None)
        mock_status.return_value.__exit__ = Mock(return_value=None)

        mock_remote = Mock(spec=BasetenRemote)
        mock_api = Mock()
        mock_remote.api = mock_api
        mock_api.get_teams.return_value = {"Team Alpha": {"id": "team1", "name": "Team Alpha"}}
        mock_remote_factory.return_value = mock_remote

        mock_create_job.return_value = {
            "id": "job123",
            "training_project": {"id": "12345", "name": "test-project"},
        }

        runner = CliRunner()
        config_path = Path("/tmp/test_config.py")
        # Create a dummy config file for the test
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text("# dummy config")

        result = runner.invoke(
            truss_cli,
            [
                "train",
                "push",
                str(config_path),
                "--remote",
                "test_remote",
                "--team",
                "Team Alpha",
            ],
        )

        assert result.exit_code == 0
        mock_create_job.assert_called_once()
        call_args = mock_create_job.call_args
        assert call_args[1]["team_id"] == "team1"

    @patch("truss_train.deployment.create_training_job_from_file")
    @patch("truss.cli.train_commands.RemoteFactory.create")
    @patch("truss.cli.remote_cli.inquire_team")
    @patch("truss.cli.train_commands.console.status")
    @patch("truss.cli.train_commands._handle_post_create_logic")
    def test_team_not_provided_inquire_team_called(
        self,
        mock_post_create,
        mock_status,
        mock_inquire_team,
        mock_remote_factory,
        mock_create_job,
    ):
        """Test that inquire_team is called when --team is not provided."""
        mock_status.return_value.__enter__ = Mock(return_value=None)
        mock_status.return_value.__exit__ = Mock(return_value=None)

        mock_remote = Mock(spec=BasetenRemote)
        mock_api = Mock()
        mock_remote.api = mock_api
        mock_api.get_teams.return_value = {
            "Team Alpha": {"id": "team1", "name": "Team Alpha"},
            "Team Beta": {"id": "team2", "name": "Team Beta"},
        }
        mock_remote_factory.return_value = mock_remote

        mock_inquire_team.return_value = "team2"
        mock_create_job.return_value = {
            "id": "job123",
            "training_project": {"id": "12345", "name": "test-project"},
        }

        runner = CliRunner()
        config_path = Path("/tmp/test_config.py")
        # Create a dummy config file for the test
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text("# dummy config")

        result = runner.invoke(
            truss_cli, ["train", "push", str(config_path), "--remote", "test_remote"]
        )

        assert result.exit_code == 0
        # inquire_team should be called with remote_provider and teams parameter
        mock_inquire_team.assert_called_once()
        call_args = mock_inquire_team.call_args
        assert call_args[0][0] == mock_remote
        assert call_args[1]["teams"] == {
            "Team Alpha": {"id": "team1", "name": "Team Alpha"},
            "Team Beta": {"id": "team2", "name": "Team Beta"},
        }
        mock_create_job.assert_called_once()
        call_args = mock_create_job.call_args
        assert call_args[1]["team_id"] == "team2"
