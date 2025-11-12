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
        mock_remote = Mock(spec=BasetenRemote)
        mock_api = Mock()
        mock_remote.api = mock_api
        mock_api.get_teams.return_value = teams
        return mock_remote

    @staticmethod
    def _create_test_config():
        config_path = Path("/tmp/test_config.py")
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text("# dummy config")
        return config_path

    @staticmethod
    def _invoke_train_push(runner, config_path, team_name=None, remote="test_remote"):
        args = ["train", "push", str(config_path), "--remote", remote]
        if team_name:
            args.extend(["--team", team_name])
        return runner.invoke(truss_cli, args)

    @staticmethod
    def _create_mock_training_project(name="test-project"):
        mock_project = Mock()
        mock_project.name = name
        return mock_project

    @staticmethod
    def _setup_mock_loader(mock_import_project, training_project):
        mock_import_project.return_value.__enter__ = Mock(return_value=training_project)
        mock_import_project.return_value.__exit__ = Mock(return_value=None)

    @staticmethod
    def _setup_mock_status(mock_status):
        mock_status.return_value.__enter__ = Mock(return_value=None)
        mock_status.return_value.__exit__ = Mock(return_value=None)

    @staticmethod
    def _create_mock_job_response(
        project_id="12345", project_name="test-project", job_id="job123"
    ):
        return {
            "id": job_id,
            "training_project": {"id": project_id, "name": project_name},
        }

    @staticmethod
    def _assert_training_job_called_with_team(
        mock_create_job, expected_team_id, training_project
    ):
        mock_create_job.assert_called_once()
        call_args = mock_create_job.call_args
        assert call_args[0][2] == training_project
        assert call_args[1]["team_id"] == expected_team_id

    @patch("truss_train.deployment.create_training_job")
    @patch("truss.cli.train_commands.RemoteFactory.create")
    @patch("truss.cli.train_commands.console.status")
    @patch("truss_train.loader.import_training_project")
    def test_team_not_provided_single_team_uses_default(
        self, mock_import_project, mock_status, mock_remote_factory, mock_create_job
    ):
        """Test that default team is used when --team not provided and user has 1 team."""
        teams = {"Team Alpha": {"id": "team1", "name": "Team Alpha"}}
        training_project = self._create_mock_training_project()
        job_response = self._create_mock_job_response()

        mock_remote = self._setup_mock_remote(teams)
        mock_remote.api.list_training_projects.return_value = []
        mock_remote_factory.return_value = mock_remote
        self._setup_mock_status(mock_status)
        self._setup_mock_loader(mock_import_project, training_project)
        mock_create_job.return_value = job_response

        runner = CliRunner()
        config_path = self._create_test_config()
        result = self._invoke_train_push(runner, config_path)

        assert result.exit_code == 0
        self._assert_training_job_called_with_team(
            mock_create_job, "team1", training_project
        )

    @patch("truss_train.deployment.create_training_job")
    @patch("truss.cli.train_commands.RemoteFactory.create")
    @patch("truss.cli.remote_cli.inquire_team")
    @patch("truss.cli.train_commands.console.status")
    @patch("truss_train.loader.import_training_project")
    def test_team_not_provided_multiple_teams_inquires(
        self,
        mock_import_project,
        mock_status,
        mock_inquire_team,
        mock_remote_factory,
        mock_create_job,
    ):
        """Test that inquire_team is called when --team not provided and user has > 1 team."""
        teams = {
            "Team Alpha": {"id": "team1", "name": "Team Alpha"},
            "Team Beta": {"id": "team2", "name": "Team Beta"},
        }
        training_project = self._create_mock_training_project()
        job_response = self._create_mock_job_response()

        mock_remote = self._setup_mock_remote(teams)
        mock_remote.api.list_training_projects.return_value = []
        mock_remote_factory.return_value = mock_remote
        self._setup_mock_status(mock_status)
        self._setup_mock_loader(mock_import_project, training_project)
        mock_inquire_team.return_value = "team2"
        mock_create_job.return_value = job_response

        runner = CliRunner()
        config_path = self._create_test_config()
        result = self._invoke_train_push(runner, config_path)

        assert result.exit_code == 0
        mock_inquire_team.assert_called_once()
        assert mock_inquire_team.call_args[1]["existing_teams"] == teams
        self._assert_training_job_called_with_team(
            mock_create_job, "team2", training_project
        )

    @patch("truss_train.deployment.create_training_job")
    @patch("truss.cli.train_commands.RemoteFactory.create")
    @patch("truss.cli.train_commands.console.status")
    @patch("truss_train.loader.import_training_project")
    def test_team_provided_valid_propagated_to_backend(
        self, mock_import_project, mock_status, mock_remote_factory, mock_create_job
    ):
        """Test that --team parameter is propagated to backend request when valid."""
        teams = {"Team Alpha": {"id": "team1", "name": "Team Alpha"}}
        training_project = self._create_mock_training_project()
        job_response = self._create_mock_job_response()

        mock_remote = self._setup_mock_remote(teams)
        mock_remote_factory.return_value = mock_remote
        self._setup_mock_status(mock_status)
        self._setup_mock_loader(mock_import_project, training_project)
        mock_create_job.return_value = job_response

        runner = CliRunner()
        config_path = self._create_test_config()
        result = self._invoke_train_push(runner, config_path, team_name="Team Alpha")

        assert result.exit_code == 0
        self._assert_training_job_called_with_team(
            mock_create_job, "team1", training_project
        )

    @patch("truss.cli.train_commands.RemoteFactory.create")
    @patch("truss_train.loader.import_training_project")
    def test_team_provided_invalid_raises_exception(
        self, mock_import_project, mock_remote_factory
    ):
        """Test that invalid --team raises exception and command fails."""
        teams = {"Team Alpha": {"id": "team1", "name": "Team Alpha"}}
        training_project = self._create_mock_training_project()

        mock_remote = self._setup_mock_remote(teams)
        mock_remote_factory.return_value = mock_remote
        self._setup_mock_loader(mock_import_project, training_project)

        runner = CliRunner()
        config_path = self._create_test_config()
        result = self._invoke_train_push(runner, config_path, team_name="Invalid Team")

        assert result.exit_code != 0
        assert "does not exist" in result.output
        assert "Invalid Team" in result.output
        assert "Team" in result.output and "Alpha" in result.output

    @patch("truss_train.deployment.create_training_job")
    @patch("truss.cli.train_commands.RemoteFactory.create")
    @patch("truss.cli.train_commands.console.status")
    @patch("truss_train.loader.import_training_project")
    def test_team_auto_detected_from_existing_project(
        self, mock_import_project, mock_status, mock_remote_factory, mock_create_job
    ):
        """Test that team_id is auto-detected when project exists and user has 1 team."""
        teams = {"Team Alpha": {"id": "team1", "name": "Team Alpha"}}
        existing_project = {
            "id": "project123",
            "name": "existing-project",
            "team_name": "Team Alpha",
        }
        training_project = self._create_mock_training_project(name="existing-project")
        job_response = self._create_mock_job_response(
            project_id="project123", project_name="existing-project"
        )

        mock_remote = self._setup_mock_remote(teams)
        mock_remote.api.list_training_projects.return_value = [existing_project]
        mock_remote_factory.return_value = mock_remote
        self._setup_mock_status(mock_status)
        self._setup_mock_loader(mock_import_project, training_project)
        mock_create_job.return_value = job_response

        runner = CliRunner()
        config_path = self._create_test_config()
        result = self._invoke_train_push(runner, config_path)

        assert result.exit_code == 0
        self._assert_training_job_called_with_team(
            mock_create_job, "team1", training_project
        )
        mock_remote.api.list_training_projects.assert_called_once()
