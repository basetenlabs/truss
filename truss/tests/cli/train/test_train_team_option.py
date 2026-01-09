"""Tests for --team option in truss train commands."""

from unittest.mock import Mock, patch

from click.testing import CliRunner

from truss.cli.cli import truss_cli
from truss.remote.baseten.remote import BasetenRemote


class TestTrainCommandsTeamOption:
    """Test --team option in truss train commands."""

    @staticmethod
    def _setup_mock_remote(projects):
        mock_remote = Mock(spec=BasetenRemote)
        mock_api = Mock()
        mock_remote.api = mock_api
        mock_api.list_training_projects.return_value = projects
        mock_remote.remote_url = "https://app.baseten.co"
        return mock_remote

    @patch("truss.cli.train_commands.train_cli.view_training_details")
    @patch("truss.cli.train_commands.train_cli.view_training_job_metrics")
    @patch("truss.cli.train_commands.train_cli.get_args_for_stop")
    @patch("truss.cli.train_commands.train_cli.view_cache_summary_by_project")
    @patch("truss.cli.train_commands.RemoteFactory.create")
    def test_team_option_filters_correctly(
        self,
        mock_remote_factory,
        mock_view_cache,
        mock_get_args,
        mock_view_metrics,
        mock_view_details,
    ):
        """Test that --team option filters projects correctly across commands."""
        projects = [
            {"id": "proj-123", "name": "my-project", "team_name": "Team Alpha"},
            {"id": "proj-456", "name": "my-project", "team_name": "Team Beta"},
        ]
        mock_remote = self._setup_mock_remote(projects)
        mock_remote_factory.return_value = mock_remote
        mock_get_args.return_value = ("proj-456", "job-789")

        runner = CliRunner()

        # Test truss train view --team
        runner.invoke(
            truss_cli,
            [
                "train",
                "view",
                "--project",
                "my-project",
                "--team",
                "Team Beta",
                "--remote",
                "test",
            ],
        )
        assert mock_view_details.call_args[0][1] == "proj-456"

        # Test truss train metrics --team
        runner.invoke(
            truss_cli,
            [
                "train",
                "metrics",
                "--project",
                "my-project",
                "--team",
                "Team Alpha",
                "--remote",
                "test",
            ],
        )
        assert mock_view_metrics.call_args[0][1] == "proj-123"

        # Test truss train stop --team
        runner.invoke(
            truss_cli,
            [
                "train",
                "stop",
                "--project",
                "my-project",
                "--team",
                "Team Beta",
                "--remote",
                "test",
            ],
        )
        assert mock_get_args.call_args[0][1] == "proj-456"

        # Test truss train cache summarize --team
        runner.invoke(
            truss_cli,
            [
                "train",
                "cache",
                "summarize",
                "my-project",
                "--team",
                "Team Beta",
                "--remote",
                "test",
            ],
        )
        assert mock_view_cache.call_args[1]["team_name"] == "Team Beta"

    @patch("truss.cli.train_commands.RemoteFactory.create")
    @patch("truss.cli.train.core.inquirer.select")
    def test_ambiguous_project_prompts_once(self, mock_select, mock_remote_factory):
        """Test that ambiguous project name prompts user exactly once."""
        projects = [
            {"id": "proj-123", "name": "my-project", "team_name": "Team Alpha"},
            {"id": "proj-456", "name": "my-project", "team_name": "Team Beta"},
        ]
        mock_remote = self._setup_mock_remote(projects)
        mock_remote_factory.return_value = mock_remote
        mock_select.return_value.execute.return_value = "Team Beta"

        runner = CliRunner()
        with patch("truss.cli.train_commands.train_cli.view_training_details"):
            runner.invoke(
                truss_cli,
                ["train", "view", "--project", "my-project", "--remote", "test"],
            )

        assert mock_select.call_count == 1
        assert "Team Alpha" in mock_select.call_args[1]["choices"]
        assert "Team Beta" in mock_select.call_args[1]["choices"]

    @patch("truss.cli.train_commands.RemoteFactory.create")
    def test_team_option_error_when_not_found(self, mock_remote_factory):
        """Test error when project not found in specified team."""
        projects = [{"id": "proj-123", "name": "my-project", "team_name": "Team Alpha"}]
        mock_remote = self._setup_mock_remote(projects)
        mock_remote_factory.return_value = mock_remote

        runner = CliRunner()
        result = runner.invoke(
            truss_cli,
            [
                "train",
                "view",
                "--project",
                "my-project",
                "--team",
                "Team Beta",
                "--remote",
                "test",
            ],
        )

        assert result.exit_code != 0
        assert "not found" in result.output
        assert "Team Beta" in result.output
