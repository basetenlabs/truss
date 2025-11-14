"""Tests for truss chains CLI commands."""

from pathlib import Path
from unittest.mock import Mock, patch

from click.testing import CliRunner

from truss.cli.cli import truss_cli
from truss.remote.baseten.remote import BasetenRemote


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
    runner = CliRunner()

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
                    "--publish",
                    "--dryrun",
                ],
            )

    assert result.exit_code == 0

    mock_push.assert_called_once()
    call_args = mock_push.call_args
    options = call_args[0][1]

    assert hasattr(options, "deployment_name")
    assert options.deployment_name == "custom_deployment"


class TestChainsTeamParameter:
    """Test team parameter in chains push command."""

    @staticmethod
    def _create_mock_entrypoint():
        """Create a mock entrypoint class."""
        mock_entrypoint_cls = Mock()
        mock_entrypoint_cls.meta_data.chain_name = "test_chain"
        mock_entrypoint_cls.display_name = "TestChain"
        return mock_entrypoint_cls

    @staticmethod
    def _create_mock_service():
        """Create a mock BasetenChainService."""
        mock_service = Mock()
        mock_service.run_remote_url = "http://test.com/run_remote"
        mock_service.is_websocket = False
        mock_service.status_page_url = "http://test.com/status"
        mock_service.name = "test_chain"
        # Mock get_info() to return a list with one chainlet info
        mock_chainlet_info = Mock()
        mock_chainlet_info.is_entrypoint = True
        mock_chainlet_info.name = "TestChain"
        mock_chainlet_info.status = "ACTIVE"
        mock_chainlet_info.logs_url = "http://test.com/logs"
        mock_service.get_info.return_value = [mock_chainlet_info]
        return mock_service

    @staticmethod
    def _create_mock_remote(teams):
        """Create a mock BasetenRemote with specified teams."""
        mock_remote = Mock(spec=BasetenRemote)
        mock_api = Mock()
        mock_remote.api = mock_api
        mock_api.get_teams.return_value = teams
        return mock_remote

    @staticmethod
    def _create_test_chain_file():
        """Create a temporary test chain file."""
        chain_file = Path("/tmp/test_chain.py")
        chain_file.parent.mkdir(parents=True, exist_ok=True)
        chain_file.write_text("# dummy chain file")
        return chain_file

    @staticmethod
    def _mock_isinstance_check(obj, cls):
        """Mock isinstance to return True for BasetenChainService check."""
        if hasattr(cls, "__name__") and cls.__name__ == "BasetenChainService":
            return True
        return isinstance(obj, cls)

    @patch("truss_chains.framework.ChainletImporter.import_target")
    @patch("truss_chains.deployment.deployment_client.push")
    @patch("truss.cli.chains_commands.RemoteFactory.create")
    @patch("truss.cli.chains_commands.isinstance")
    def test_team_not_provided_single_team_defaults(
        self, mock_isinstance, mock_remote_factory, mock_push, mock_importer
    ):
        """Test that when --team is not provided and user has 1 team, default team is used."""
        mock_entrypoint_cls = self._create_mock_entrypoint()
        mock_service = self._create_mock_service()
        mock_remote = self._create_mock_remote(
            {"Team Alpha": {"id": "team1", "name": "Team Alpha"}}
        )

        mock_importer.return_value.__enter__.return_value = mock_entrypoint_cls
        mock_push.return_value = mock_service
        mock_remote_factory.return_value = mock_remote
        mock_isinstance.side_effect = self._mock_isinstance_check

        chain_file = self._create_test_chain_file()
        runner = CliRunner()

        result = runner.invoke(
            truss_cli,
            ["chains", "push", str(chain_file), "--remote", "test_remote", "--no-wait"],
        )

        assert result.exit_code == 0
        mock_push.assert_called_once()
        call_args = mock_push.call_args
        options = call_args[0][1]
        assert options.team_id == "team1"

    @patch("truss_chains.framework.ChainletImporter.import_target")
    @patch("truss_chains.deployment.deployment_client.push")
    @patch("truss.cli.remote_cli.inquire_team")
    @patch("truss.cli.chains_commands.RemoteFactory.create")
    @patch("truss.cli.chains_commands.isinstance")
    def test_team_not_provided_multiple_teams_inquires(
        self,
        mock_isinstance,
        mock_remote_factory,
        mock_inquire_team,
        mock_push,
        mock_importer,
    ):
        """Test that when --team is not provided and user has >1 teams, inquire_team is called."""
        mock_entrypoint_cls = self._create_mock_entrypoint()
        mock_service = self._create_mock_service()
        teams = {
            "Team Alpha": {"id": "team1", "name": "Team Alpha"},
            "Team Beta": {"id": "team2", "name": "Team Beta"},
        }
        mock_remote = self._create_mock_remote(teams)

        mock_importer.return_value.__enter__.return_value = mock_entrypoint_cls
        mock_push.return_value = mock_service
        mock_remote_factory.return_value = mock_remote
        mock_inquire_team.return_value = "team2"
        mock_isinstance.side_effect = self._mock_isinstance_check

        chain_file = self._create_test_chain_file()
        runner = CliRunner()

        result = runner.invoke(
            truss_cli,
            ["chains", "push", str(chain_file), "--remote", "test_remote", "--no-wait"],
        )

        assert result.exit_code == 0
        mock_inquire_team.assert_called_once()
        call_args = mock_inquire_team.call_args
        assert call_args[1]["existing_teams"] == teams
        mock_push.assert_called_once()
        call_args = mock_push.call_args
        options = call_args[0][1]
        assert options.team_id == "team2"

    @patch("truss_chains.framework.ChainletImporter.import_target")
    @patch("truss_chains.deployment.deployment_client.push")
    @patch("truss.cli.chains_commands.RemoteFactory.create")
    @patch("truss.cli.chains_commands.isinstance")
    def test_team_provided_valid_propagates(
        self, mock_isinstance, mock_remote_factory, mock_push, mock_importer
    ):
        """Test that when --team is provided and valid, team_id is propagated."""
        mock_entrypoint_cls = self._create_mock_entrypoint()
        mock_service = self._create_mock_service()
        mock_remote = self._create_mock_remote(
            {
                "Team Alpha": {"id": "team1", "name": "Team Alpha"},
                "Team Beta": {"id": "team2", "name": "Team Beta"},
            }
        )

        mock_importer.return_value.__enter__.return_value = mock_entrypoint_cls
        mock_push.return_value = mock_service
        mock_remote_factory.return_value = mock_remote
        mock_isinstance.side_effect = self._mock_isinstance_check

        chain_file = self._create_test_chain_file()
        runner = CliRunner()

        result = runner.invoke(
            truss_cli,
            [
                "chains",
                "push",
                str(chain_file),
                "--remote",
                "test_remote",
                "--team",
                "Team Alpha",
                "--no-wait",
            ],
        )

        assert result.exit_code == 0
        mock_push.assert_called_once()
        call_args = mock_push.call_args
        options = call_args[0][1]
        assert options.team_id == "team1"

    @patch("truss_chains.framework.ChainletImporter.import_target")
    @patch("truss.cli.chains_commands.RemoteFactory.create")
    def test_team_provided_invalid_raises_exception(
        self, mock_remote_factory, mock_importer
    ):
        """Test that when --team is provided but invalid, an exception is raised."""
        mock_entrypoint_cls = self._create_mock_entrypoint()
        mock_remote = self._create_mock_remote(
            {
                "Team Alpha": {"id": "team1", "name": "Team Alpha"},
                "Team Beta": {"id": "team2", "name": "Team Beta"},
            }
        )

        mock_importer.return_value.__enter__.return_value = mock_entrypoint_cls
        mock_remote_factory.return_value = mock_remote

        chain_file = self._create_test_chain_file()
        runner = CliRunner()

        result = runner.invoke(
            truss_cli,
            [
                "chains",
                "push",
                str(chain_file),
                "--remote",
                "test_remote",
                "--team",
                "Invalid Team",
                "--no-wait",
            ],
        )

        assert result.exit_code != 0
        assert "does not exist" in result.output or "Invalid Team" in result.output
