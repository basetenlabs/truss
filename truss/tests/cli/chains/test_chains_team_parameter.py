"""Tests for team parameter in chain deployment.

This test suite covers all 8 scenarios for team resolution in truss chains push:
1. --team PROVIDED: Valid team name, user has access, user has 1 existing team
2. --team PROVIDED: Valid team name, user has access, user has multiple existing teams
3. --team PROVIDED: Invalid team name (does not exist)
4. --team NOT PROVIDED: User has multiple teams, no existing chain
5. --team NOT PROVIDED: User has multiple teams, existing chain exists in multiple teams
6. --team NOT PROVIDED: User has multiple teams, existing chain in exactly one team
7. --team NOT PROVIDED: User has exactly one team, no existing chain
8. --team NOT PROVIDED: User has exactly one team, existing chain matches the team
"""

from pathlib import Path
from unittest.mock import Mock, patch

from click.testing import CliRunner

from truss.cli.cli import truss_cli
from truss.remote.baseten.remote import BasetenRemote


class TestChainsTeamParameter:
    """Test team parameter in chain deployment."""

    @staticmethod
    def _setup_mock_remote(teams):
        mock_remote = Mock(spec=BasetenRemote)
        mock_api = Mock()
        mock_remote.api = mock_api
        mock_api.get_teams.return_value = teams
        return mock_remote

    @staticmethod
    def _create_test_chain_file():
        chain_path = Path("/tmp/test_chain.py")
        chain_path.parent.mkdir(parents=True, exist_ok=True)
        chain_path.write_text(
            """
from truss_chains import Chainlet, mark_entrypoint

@mark_entrypoint
class TestChain(Chainlet[str, str]):
    def run(self, inp: str) -> str:
        return inp
"""
        )
        return chain_path

    @staticmethod
    def _invoke_chains_push(runner, chain_path, team_name=None, remote="test_remote"):
        args = ["chains", "push", str(chain_path), "--remote", remote]
        if team_name:
            args.extend(["--team", team_name])
        return runner.invoke(truss_cli, args)

    @staticmethod
    def _create_mock_chainlet():
        mock_chainlet = Mock()
        mock_meta_data = Mock()
        mock_meta_data.chain_name = None
        mock_chainlet.meta_data = mock_meta_data
        mock_chainlet.display_name = "TestChain"
        return mock_chainlet

    @staticmethod
    def _setup_mock_chainlet_importer(mock_import_target, chainlet):
        context_manager = Mock()
        context_manager.__enter__ = Mock(return_value=chainlet)
        context_manager.__exit__ = Mock(return_value=None)
        mock_import_target.return_value = context_manager

    @staticmethod
    def _create_mock_chain_service():
        mock_service = Mock()
        mock_service.name = "TestChain"
        mock_service.status_page_url = "https://app.baseten.co/chains/test123/overview"
        mock_service.run_remote_url = "https://app.baseten.co/chains/test123/run_remote"
        mock_service.is_websocket = False
        # Create a mock chainlet info object
        mock_chainlet_info = Mock()
        mock_chainlet_info.is_entrypoint = True
        mock_chainlet_info.name = "TestChain"
        mock_chainlet_info.status = "ACTIVE"
        mock_chainlet_info.logs_url = "https://app.baseten.co/chains/test123/logs"
        mock_service.get_info.return_value = [mock_chainlet_info]
        return mock_service

    @staticmethod
    def _assert_push_called_with_team(
        mock_push, expected_team_id, expected_chain_name, expected_teams=None
    ):
        mock_push.assert_called_once()
        call_args = mock_push.call_args
        options = call_args[0][1]  # Second argument is options
        assert options.chain_name == expected_chain_name
        assert options.team_id == expected_team_id

    # SCENARIO 1: --team PROVIDED: Valid team name, user has access, user has 1 existing team
    # CLI Command: truss chains push /path/to/chain.py --team "Team Alpha" --remote baseten_staging
    # Exit Code: 0, Error Message: None, Interactive Prompt: No, Existing Teams: ["team1"]
    @patch("truss_chains.deployment.deployment_client.push")
    @patch("truss.cli.chains_commands.RemoteFactory.create")
    @patch("truss_chains.framework.ChainletImporter.import_target")
    def test_scenario_1_team_provided_valid_team_name_single_team(
        self, mock_import_target, mock_remote_factory, mock_push
    ):
        """Scenario 1: --team PROVIDED with valid team name, user has 1 existing team."""
        teams = {"Team Alpha": {"id": "team1", "name": "Team Alpha"}}
        chainlet = self._create_mock_chainlet()
        chain_service = self._create_mock_chain_service()

        mock_remote = self._setup_mock_remote(teams)
        mock_remote_factory.return_value = mock_remote
        self._setup_mock_chainlet_importer(mock_import_target, chainlet)
        mock_push.return_value = chain_service

        runner = CliRunner()
        chain_path = self._create_test_chain_file()
        result = self._invoke_chains_push(runner, chain_path, team_name="Team Alpha")

        assert result.exit_code == 0
        self._assert_push_called_with_team(
            mock_push, "team1", "TestChain", expected_teams=teams
        )

    # SCENARIO 2: --team PROVIDED: Valid team name, user has access, user has multiple existing teams
    # CLI Command: truss chains push /path/to/chain.py --team "Team Alpha" --remote baseten_staging
    # Exit Code: 0, Error Message: None, Interactive Prompt: No, Existing Teams: ["team1", "team2", "team3"]
    @patch("truss_chains.deployment.deployment_client.push")
    @patch("truss.cli.chains_commands.RemoteFactory.create")
    @patch("truss_chains.framework.ChainletImporter.import_target")
    def test_scenario_2_team_provided_valid_team_name_multiple_teams(
        self, mock_import_target, mock_remote_factory, mock_push
    ):
        """Scenario 2: --team PROVIDED with valid team name, user has multiple teams."""
        teams = {
            "Team Alpha": {"id": "team1", "name": "Team Alpha"},
            "Team Beta": {"id": "team2", "name": "Team Beta"},
            "Team Gamma": {"id": "team3", "name": "Team Gamma"},
        }
        chainlet = self._create_mock_chainlet()
        chain_service = self._create_mock_chain_service()

        mock_remote = self._setup_mock_remote(teams)
        mock_remote_factory.return_value = mock_remote
        self._setup_mock_chainlet_importer(mock_import_target, chainlet)
        mock_push.return_value = chain_service

        runner = CliRunner()
        chain_path = self._create_test_chain_file()
        result = self._invoke_chains_push(runner, chain_path, team_name="Team Alpha")

        assert result.exit_code == 0
        self._assert_push_called_with_team(
            mock_push, "team1", "TestChain", expected_teams=teams
        )

    # SCENARIO 3: --team PROVIDED: Invalid team name (does not exist)
    # CLI Command: truss chains push /path/to/chain.py --team "NonExistentTeam" --remote baseten_staging
    # Exit Code: 1, Error Message: Team does not exist, Interactive Prompt: No, Existing Teams: ["team1"]
    @patch("truss.cli.chains_commands.RemoteFactory.create")
    @patch("truss_chains.framework.ChainletImporter.import_target")
    def test_scenario_3_team_provided_invalid_team_name(
        self, mock_import_target, mock_remote_factory
    ):
        """Scenario 3: --team PROVIDED with invalid team name that does not exist."""
        teams = {"Team Alpha": {"id": "team1", "name": "Team Alpha"}}
        chainlet = self._create_mock_chainlet()

        mock_remote = self._setup_mock_remote(teams)
        mock_remote_factory.return_value = mock_remote
        self._setup_mock_chainlet_importer(mock_import_target, chainlet)

        runner = CliRunner()
        chain_path = self._create_test_chain_file()
        result = self._invoke_chains_push(
            runner, chain_path, team_name="NonExistentTeam"
        )

        assert result.exit_code == 1
        assert "does not exist" in result.output
        assert "NonExistentTeam" in result.output

    # SCENARIO 4: --team NOT PROVIDED: User has multiple teams, no existing chain
    # CLI Command: truss chains push /path/to/chain.py --remote baseten_staging
    # Exit Code: 0, Error Message: None, Interactive Prompt: Yes, Existing Teams: ["team1", "team2", "team3"]
    @patch("truss_chains.deployment.deployment_client.push")
    @patch("truss.cli.chains_commands.RemoteFactory.create")
    @patch("truss.cli.remote_cli.inquire_team")
    @patch("truss_chains.framework.ChainletImporter.import_target")
    def test_scenario_4_multiple_teams_no_existing_chain(
        self, mock_import_target, mock_inquire_team, mock_remote_factory, mock_push
    ):
        """Scenario 4: --team NOT PROVIDED, user has multiple teams, no existing chain."""
        teams = {
            "Team Alpha": {"id": "team1", "name": "Team Alpha"},
            "Team Beta": {"id": "team2", "name": "Team Beta"},
            "Team Gamma": {"id": "team3", "name": "Team Gamma"},
        }
        chainlet = self._create_mock_chainlet()
        chain_service = self._create_mock_chain_service()

        mock_remote = self._setup_mock_remote(teams)
        mock_remote.api.get_chains.return_value = []
        mock_remote_factory.return_value = mock_remote
        self._setup_mock_chainlet_importer(mock_import_target, chainlet)
        mock_inquire_team.return_value = "Team Beta"
        mock_push.return_value = chain_service

        runner = CliRunner()
        chain_path = self._create_test_chain_file()
        result = self._invoke_chains_push(runner, chain_path)

        assert result.exit_code == 0
        mock_inquire_team.assert_called_once()
        assert mock_inquire_team.call_args[1]["existing_teams"] == teams
        self._assert_push_called_with_team(
            mock_push, "team2", "TestChain", expected_teams=teams
        )

    # SCENARIO 5: --team NOT PROVIDED: User has multiple teams, existing chain exists in multiple teams
    # CLI Command: truss chains push /path/to/chain.py --remote baseten_staging
    # Exit Code: 0, Error Message: None, Interactive Prompt: Yes, Existing Teams: ["team1", "team2", "team3"]
    @patch("truss_chains.deployment.deployment_client.push")
    @patch("truss.cli.chains_commands.RemoteFactory.create")
    @patch("truss.cli.remote_cli.inquire_team")
    @patch("truss_chains.framework.ChainletImporter.import_target")
    def test_scenario_5_multiple_teams_existing_chain_in_multiple_teams(
        self, mock_import_target, mock_inquire_team, mock_remote_factory, mock_push
    ):
        """Scenario 5: --team NOT PROVIDED, multiple teams, existing chain in multiple teams."""
        teams = {
            "Team Alpha": {"id": "team1", "name": "Team Alpha"},
            "Team Beta": {"id": "team2", "name": "Team Beta"},
            "Team Gamma": {"id": "team3", "name": "Team Gamma"},
        }
        existing_chains = [
            {"id": "chain123", "name": "TestChain", "team": {"name": "Team Alpha"}},
            {"id": "chain456", "name": "TestChain", "team": {"name": "Team Beta"}},
        ]
        chainlet = self._create_mock_chainlet()
        chain_service = self._create_mock_chain_service()

        mock_remote = self._setup_mock_remote(teams)
        mock_remote.api.get_chains.return_value = existing_chains
        mock_remote_factory.return_value = mock_remote
        self._setup_mock_chainlet_importer(mock_import_target, chainlet)
        mock_inquire_team.return_value = "Team Alpha"
        mock_push.return_value = chain_service

        runner = CliRunner()
        chain_path = self._create_test_chain_file()
        result = self._invoke_chains_push(runner, chain_path)

        assert result.exit_code == 0
        mock_inquire_team.assert_called_once()
        assert mock_inquire_team.call_args[1]["existing_teams"] == teams
        self._assert_push_called_with_team(
            mock_push, "team1", "TestChain", expected_teams=teams
        )

    # SCENARIO 6: --team NOT PROVIDED: User has multiple teams, existing chain in exactly one team
    # CLI Command: truss chains push /path/to/chain.py --remote baseten_staging
    # Exit Code: 0, Error Message: None, Interactive Prompt: No, Existing Teams: ["team1", "team2", "team3"]
    @patch("truss_chains.deployment.deployment_client.push")
    @patch("truss.cli.chains_commands.RemoteFactory.create")
    @patch("truss_chains.framework.ChainletImporter.import_target")
    def test_scenario_6_multiple_teams_existing_chain_in_one_team(
        self, mock_import_target, mock_remote_factory, mock_push
    ):
        """Scenario 6: --team NOT PROVIDED, multiple teams, existing chain in exactly one team."""
        teams = {
            "Team Alpha": {"id": "team1", "name": "Team Alpha"},
            "Team Beta": {"id": "team2", "name": "Team Beta"},
            "Team Gamma": {"id": "team3", "name": "Team Gamma"},
        }
        existing_chain = {
            "id": "chain123",
            "name": "TestChain",
            "team": {"name": "Team Beta"},
        }
        chainlet = self._create_mock_chainlet()
        chain_service = self._create_mock_chain_service()

        mock_remote = self._setup_mock_remote(teams)
        mock_remote.api.get_chains.return_value = [existing_chain]
        mock_remote_factory.return_value = mock_remote
        self._setup_mock_chainlet_importer(mock_import_target, chainlet)
        mock_push.return_value = chain_service

        runner = CliRunner()
        chain_path = self._create_test_chain_file()
        result = self._invoke_chains_push(runner, chain_path)

        assert result.exit_code == 0
        self._assert_push_called_with_team(
            mock_push, "team2", "TestChain", expected_teams=teams
        )
        mock_remote.api.get_chains.assert_called()

    # SCENARIO 7: --team NOT PROVIDED: User has exactly one team, no existing chain
    # CLI Command: truss chains push /path/to/chain.py --remote baseten_staging
    # Exit Code: 0, Error Message: None, Interactive Prompt: No, Existing Teams: ["team1"]
    @patch("truss_chains.deployment.deployment_client.push")
    @patch("truss.cli.chains_commands.RemoteFactory.create")
    @patch("truss_chains.framework.ChainletImporter.import_target")
    def test_scenario_7_single_team_no_existing_chain(
        self, mock_import_target, mock_remote_factory, mock_push
    ):
        """Scenario 7: --team NOT PROVIDED, user has exactly one team, no existing chain."""
        teams = {"Team Alpha": {"id": "team1", "name": "Team Alpha"}}
        chainlet = self._create_mock_chainlet()
        chain_service = self._create_mock_chain_service()

        mock_remote = self._setup_mock_remote(teams)
        mock_remote.api.get_chains.return_value = []
        mock_remote_factory.return_value = mock_remote
        self._setup_mock_chainlet_importer(mock_import_target, chainlet)
        mock_push.return_value = chain_service

        runner = CliRunner()
        chain_path = self._create_test_chain_file()
        result = self._invoke_chains_push(runner, chain_path)

        assert result.exit_code == 0
        self._assert_push_called_with_team(
            mock_push, "team1", "TestChain", expected_teams=teams
        )

    # SCENARIO 8: --team NOT PROVIDED: User has exactly one team, existing chain matches the team
    # CLI Command: truss chains push /path/to/chain.py --remote baseten_staging
    # Exit Code: 0, Error Message: None, Interactive Prompt: No, Existing Teams: ["team1"]
    @patch("truss_chains.deployment.deployment_client.push")
    @patch("truss.cli.chains_commands.RemoteFactory.create")
    @patch("truss_chains.framework.ChainletImporter.import_target")
    def test_scenario_8_single_team_existing_chain_matches_team(
        self, mock_import_target, mock_remote_factory, mock_push
    ):
        """Scenario 8: --team NOT PROVIDED, single team, existing chain matches the team."""
        teams = {"Team Alpha": {"id": "team1", "name": "Team Alpha"}}
        existing_chain = {
            "id": "chain123",
            "name": "TestChain",
            "team": {"name": "Team Alpha"},
        }
        chainlet = self._create_mock_chainlet()
        chain_service = self._create_mock_chain_service()

        mock_remote = self._setup_mock_remote(teams)
        mock_remote.api.get_chains.return_value = [existing_chain]
        mock_remote_factory.return_value = mock_remote
        self._setup_mock_chainlet_importer(mock_import_target, chainlet)
        mock_push.return_value = chain_service

        runner = CliRunner()
        chain_path = self._create_test_chain_file()
        result = self._invoke_chains_push(runner, chain_path)

        assert result.exit_code == 0
        self._assert_push_called_with_team(
            mock_push, "team1", "TestChain", expected_teams=teams
        )
        mock_remote.api.get_chains.assert_called()
