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
from truss.remote.baseten.custom_types import TeamType
from truss.remote.baseten.remote import BasetenRemote


class TestChainsTeamParameter:
    """Test team parameter in chain deployment using Given-When-Then format."""

    @staticmethod
    def _given_mock_remote(teams):
        """Given: A mock remote provider with specified teams."""
        mock_remote = Mock(spec=BasetenRemote)
        mock_api = Mock()
        mock_remote.api = mock_api
        # Convert dictionaries to TeamType objects
        teams_with_type = {
            name: TeamType(**team_data) for name, team_data in teams.items()
        }
        mock_api.get_teams.return_value = teams_with_type
        return mock_remote

    @staticmethod
    def _given_test_chain_file():
        """Given: A test chain file."""
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
    def _given_mock_chainlet():
        """Given: A mock chainlet."""
        mock_chainlet = Mock()
        mock_meta_data = Mock()
        mock_meta_data.chain_name = None
        mock_chainlet.meta_data = mock_meta_data
        mock_chainlet.display_name = "TestChain"
        return mock_chainlet

    @staticmethod
    def _given_mock_chainlet_importer(mock_import_target, chainlet):
        """Given: A mock chainlet importer context manager."""
        context_manager = Mock()
        context_manager.__enter__ = Mock(return_value=chainlet)
        context_manager.__exit__ = Mock(return_value=None)
        mock_import_target.return_value = context_manager

    @staticmethod
    def _given_mock_chain_service():
        """Given: A mock chain service."""
        mock_service = Mock()
        mock_service.name = "TestChain"
        mock_service.status_page_url = "https://app.baseten.co/chains/test123/overview"
        mock_service.run_remote_url = "https://app.baseten.co/chains/test123/run_remote"
        mock_service.is_websocket = False
        mock_chainlet_info = Mock()
        mock_chainlet_info.is_entrypoint = True
        mock_chainlet_info.name = "TestChain"
        mock_chainlet_info.status = "ACTIVE"
        mock_chainlet_info.logs_url = "https://app.baseten.co/chains/test123/logs"
        mock_service.get_info.return_value = [mock_chainlet_info]
        return mock_service

    @staticmethod
    def _when_invoke_chains_push(
        runner, chain_path, team_name=None, remote="test_remote"
    ):
        """When: Invoking the chains push command."""
        args = ["chains", "push", str(chain_path), "--remote", remote]
        if team_name:
            args.extend(["--team", team_name])
        return runner.invoke(truss_cli, args)

    @staticmethod
    def _then_assert_push_called_with_team(
        mock_push, expected_team_id, expected_chain_name
    ):
        """Then: Assert push was called with correct team_id."""
        mock_push.assert_called_once()
        call_args = mock_push.call_args
        options = call_args[0][1]  # Second argument is options
        assert options.chain_name == expected_chain_name
        assert options.team_id == expected_team_id

    def _patch_isinstance_for_mock_service(self, chain_service):
        """Helper to patch isinstance for mock service."""
        from truss_chains.deployment import deployment_client

        baseten_service_class = deployment_client.BasetenChainService

        import builtins

        original_isinstance = builtins.isinstance

        def mock_isinstance(obj, cls):
            if obj is chain_service and cls == baseten_service_class:
                return True
            return original_isinstance(obj, cls)

        builtins.isinstance = mock_isinstance
        return original_isinstance

    def _restore_isinstance(self, original_isinstance):
        """Helper to restore original isinstance."""
        import builtins

        builtins.isinstance = original_isinstance

    @patch("truss_chains.deployment.deployment_client.push")
    @patch("truss.cli.chains_commands.RemoteFactory.create")
    @patch("truss_chains.framework.ChainletImporter.import_target")
    def test_scenario_1_team_provided_valid_team_name_single_team(
        self, mock_import_target, mock_remote_factory, mock_push
    ):
        """
        Given: User has 1 team ("Team Alpha") with id "team1"
        When: User runs chains push with --team "Team Alpha"
        Then: Chain is deployed with team_id="team1" and exit code 0
        """
        teams = {"Team Alpha": {"id": "team1", "name": "Team Alpha", "default": True}}
        chainlet = self._given_mock_chainlet()
        chain_service = self._given_mock_chain_service()

        mock_remote = self._given_mock_remote(teams)
        mock_remote_factory.return_value = mock_remote
        self._given_mock_chainlet_importer(mock_import_target, chainlet)
        mock_push.return_value = chain_service

        original_isinstance = self._patch_isinstance_for_mock_service(chain_service)

        try:
            runner = CliRunner()
            chain_path = self._given_test_chain_file()
            result = self._when_invoke_chains_push(
                runner, chain_path, team_name="Team Alpha"
            )

            assert result.exit_code == 0, (
                f"Expected exit code 0, got {result.exit_code}. Output: {result.output}"
            )
            self._then_assert_push_called_with_team(mock_push, "team1", "TestChain")
        finally:
            self._restore_isinstance(original_isinstance)

    @patch("truss_chains.deployment.deployment_client.push")
    @patch("truss.cli.chains_commands.RemoteFactory.create")
    @patch("truss_chains.framework.ChainletImporter.import_target")
    def test_scenario_2_team_provided_valid_team_name_multiple_teams(
        self, mock_import_target, mock_remote_factory, mock_push
    ):
        """
        Given: User has 3 teams with "Team Alpha" having id "team1"
        When: User runs chains push with --team "Team Alpha"
        Then: Chain is deployed with team_id="team1" and exit code 0
        """
        teams = {
            "Team Alpha": {"id": "team1", "name": "Team Alpha", "default": True},
            "Team Beta": {"id": "team2", "name": "Team Beta", "default": False},
            "Team Gamma": {"id": "team3", "name": "Team Gamma", "default": False},
        }
        chainlet = self._given_mock_chainlet()
        chain_service = self._given_mock_chain_service()

        mock_remote = self._given_mock_remote(teams)
        mock_remote_factory.return_value = mock_remote
        self._given_mock_chainlet_importer(mock_import_target, chainlet)
        mock_push.return_value = chain_service

        original_isinstance = self._patch_isinstance_for_mock_service(chain_service)

        try:
            runner = CliRunner()
            chain_path = self._given_test_chain_file()
            result = self._when_invoke_chains_push(
                runner, chain_path, team_name="Team Alpha"
            )

            assert result.exit_code == 0, (
                f"Expected exit code 0, got {result.exit_code}. Output: {result.output}"
            )
            self._then_assert_push_called_with_team(mock_push, "team1", "TestChain")
        finally:
            self._restore_isinstance(original_isinstance)

    @patch("truss.cli.chains_commands.RemoteFactory.create")
    @patch("truss_chains.framework.ChainletImporter.import_target")
    def test_scenario_3_team_provided_invalid_team_name(
        self, mock_import_target, mock_remote_factory
    ):
        """
        Given: User has 1 team ("Team Alpha"), but not "NonExistentTeam"
        When: User runs chains push with --team "NonExistentTeam"
        Then: Command fails with exit code 1 and error message about team not existing
        """
        teams = {"Team Alpha": {"id": "team1", "name": "Team Alpha", "default": True}}
        chainlet = self._given_mock_chainlet()

        mock_remote = self._given_mock_remote(teams)
        mock_remote_factory.return_value = mock_remote
        self._given_mock_chainlet_importer(mock_import_target, chainlet)

        runner = CliRunner()
        chain_path = self._given_test_chain_file()
        result = self._when_invoke_chains_push(
            runner, chain_path, team_name="NonExistentTeam"
        )

        assert result.exit_code == 1
        assert "does not exist" in result.output
        assert "NonExistentTeam" in result.output

    @patch("truss_chains.deployment.deployment_client.push")
    @patch("truss.cli.chains_commands.RemoteFactory.create")
    @patch("truss.cli.remote_cli.inquire_team")
    @patch("truss_chains.framework.ChainletImporter.import_target")
    def test_scenario_4_multiple_teams_no_existing_chain(
        self, mock_import_target, mock_inquire_team, mock_remote_factory, mock_push
    ):
        """
        Given: User has 3 teams, no existing chain named "TestChain"
        When: User runs chains push without --team and selects "Team Beta"
        Then: Chain is deployed with team_id="team2" and exit code 0
        """
        teams = {
            "Team Alpha": {"id": "team1", "name": "Team Alpha", "default": True},
            "Team Beta": {"id": "team2", "name": "Team Beta", "default": False},
            "Team Gamma": {"id": "team3", "name": "Team Gamma", "default": False},
        }
        chainlet = self._given_mock_chainlet()
        chain_service = self._given_mock_chain_service()

        mock_remote = self._given_mock_remote(teams)
        mock_remote.api.get_chains.return_value = []
        mock_remote_factory.return_value = mock_remote
        self._given_mock_chainlet_importer(mock_import_target, chainlet)
        mock_inquire_team.return_value = "Team Beta"
        mock_push.return_value = chain_service

        original_isinstance = self._patch_isinstance_for_mock_service(chain_service)

        try:
            runner = CliRunner()
            chain_path = self._given_test_chain_file()
            result = self._when_invoke_chains_push(runner, chain_path)

            assert result.exit_code == 0, (
                f"Expected exit code 0, got {result.exit_code}. Output: {result.output}"
            )
            mock_inquire_team.assert_called_once()
            # Convert teams to TeamType objects for comparison
            teams_with_type = {
                name: TeamType(**team_data) for name, team_data in teams.items()
            }
            assert mock_inquire_team.call_args[1]["existing_teams"] == teams_with_type
            self._then_assert_push_called_with_team(mock_push, "team2", "TestChain")
        finally:
            self._restore_isinstance(original_isinstance)

    @patch("truss_chains.deployment.deployment_client.push")
    @patch("truss.cli.chains_commands.RemoteFactory.create")
    @patch("truss.cli.remote_cli.inquire_team")
    @patch("truss_chains.framework.ChainletImporter.import_target")
    def test_scenario_5_multiple_teams_existing_chain_in_multiple_teams(
        self, mock_import_target, mock_inquire_team, mock_remote_factory, mock_push
    ):
        """
        Given: User has 3 teams, existing chain "TestChain" in "Team Alpha" and "Team Beta"
        When: User runs chains push without --team and selects "Team Alpha"
        Then: Chain is deployed with team_id="team1" and exit code 0
        """
        teams = {
            "Team Alpha": {"id": "team1", "name": "Team Alpha", "default": True},
            "Team Beta": {"id": "team2", "name": "Team Beta", "default": False},
            "Team Gamma": {"id": "team3", "name": "Team Gamma", "default": False},
        }
        existing_chains = [
            {"id": "chain123", "name": "TestChain", "team": {"name": "Team Alpha"}},
            {"id": "chain456", "name": "TestChain", "team": {"name": "Team Beta"}},
        ]
        chainlet = self._given_mock_chainlet()
        chain_service = self._given_mock_chain_service()

        mock_remote = self._given_mock_remote(teams)
        mock_remote.api.get_chains.return_value = existing_chains
        mock_remote_factory.return_value = mock_remote
        self._given_mock_chainlet_importer(mock_import_target, chainlet)
        mock_inquire_team.return_value = "Team Alpha"
        mock_push.return_value = chain_service

        original_isinstance = self._patch_isinstance_for_mock_service(chain_service)

        try:
            runner = CliRunner()
            chain_path = self._given_test_chain_file()
            result = self._when_invoke_chains_push(runner, chain_path)

            assert result.exit_code == 0, (
                f"Expected exit code 0, got {result.exit_code}. Output: {result.output}"
            )
            mock_inquire_team.assert_called_once()
            # Convert teams to TeamType objects for comparison
            teams_with_type = {
                name: TeamType(**team_data) for name, team_data in teams.items()
            }
            assert mock_inquire_team.call_args[1]["existing_teams"] == teams_with_type
            self._then_assert_push_called_with_team(mock_push, "team1", "TestChain")
        finally:
            self._restore_isinstance(original_isinstance)

    @patch("truss_chains.deployment.deployment_client.push")
    @patch("truss.cli.chains_commands.RemoteFactory.create")
    @patch("truss_chains.framework.ChainletImporter.import_target")
    def test_scenario_6_multiple_teams_existing_chain_in_one_team(
        self, mock_import_target, mock_remote_factory, mock_push
    ):
        """
        Given: User has 3 teams, existing chain "TestChain" only in "Team Beta"
        When: User runs chains push without --team
        Then: Chain is deployed with team_id="team2" (auto-inferred) and exit code 0
        """
        teams = {
            "Team Alpha": {"id": "team1", "name": "Team Alpha", "default": True},
            "Team Beta": {"id": "team2", "name": "Team Beta", "default": False},
            "Team Gamma": {"id": "team3", "name": "Team Gamma", "default": False},
        }
        existing_chain = {
            "id": "chain123",
            "name": "TestChain",
            "team": {"name": "Team Beta"},
        }
        chainlet = self._given_mock_chainlet()
        chain_service = self._given_mock_chain_service()

        mock_remote = self._given_mock_remote(teams)
        mock_remote.api.get_chains.return_value = [existing_chain]
        mock_remote_factory.return_value = mock_remote
        self._given_mock_chainlet_importer(mock_import_target, chainlet)
        mock_push.return_value = chain_service

        original_isinstance = self._patch_isinstance_for_mock_service(chain_service)

        try:
            runner = CliRunner()
            chain_path = self._given_test_chain_file()
            result = self._when_invoke_chains_push(runner, chain_path)

            assert result.exit_code == 0, (
                f"Expected exit code 0, got {result.exit_code}. Output: {result.output}"
            )
            self._then_assert_push_called_with_team(mock_push, "team2", "TestChain")
            mock_remote.api.get_chains.assert_called()
        finally:
            self._restore_isinstance(original_isinstance)

    @patch("truss_chains.deployment.deployment_client.push")
    @patch("truss.cli.chains_commands.RemoteFactory.create")
    @patch("truss_chains.framework.ChainletImporter.import_target")
    def test_scenario_7_single_team_no_existing_chain(
        self, mock_import_target, mock_remote_factory, mock_push
    ):
        """
        Given: User has 1 team ("Team Alpha"), no existing chain
        When: User runs chains push without --team
        Then: Chain is deployed with team_id="team1" (auto-inferred) and exit code 0
        """
        teams = {"Team Alpha": {"id": "team1", "name": "Team Alpha", "default": True}}
        chainlet = self._given_mock_chainlet()
        chain_service = self._given_mock_chain_service()

        mock_remote = self._given_mock_remote(teams)
        mock_remote.api.get_chains.return_value = []
        mock_remote_factory.return_value = mock_remote
        self._given_mock_chainlet_importer(mock_import_target, chainlet)
        mock_push.return_value = chain_service

        original_isinstance = self._patch_isinstance_for_mock_service(chain_service)

        try:
            runner = CliRunner()
            chain_path = self._given_test_chain_file()
            result = self._when_invoke_chains_push(runner, chain_path)

            assert result.exit_code == 0, (
                f"Expected exit code 0, got {result.exit_code}. Output: {result.output}"
            )
            self._then_assert_push_called_with_team(mock_push, "team1", "TestChain")
        finally:
            self._restore_isinstance(original_isinstance)

    @patch("truss_chains.deployment.deployment_client.push")
    @patch("truss.cli.chains_commands.RemoteFactory.create")
    @patch("truss_chains.framework.ChainletImporter.import_target")
    def test_scenario_8_single_team_existing_chain_matches_team(
        self, mock_import_target, mock_remote_factory, mock_push
    ):
        """
        Given: User has 1 team ("Team Alpha"), existing chain "TestChain" in "Team Alpha"
        When: User runs chains push without --team
        Then: Chain is deployed with team_id="team1" (auto-inferred) and exit code 0
        """
        teams = {"Team Alpha": {"id": "team1", "name": "Team Alpha", "default": True}}
        existing_chain = {
            "id": "chain123",
            "name": "TestChain",
            "team": {"name": "Team Alpha"},
        }
        chainlet = self._given_mock_chainlet()
        chain_service = self._given_mock_chain_service()

        mock_remote = self._given_mock_remote(teams)
        mock_remote.api.get_chains.return_value = [existing_chain]
        mock_remote_factory.return_value = mock_remote
        self._given_mock_chainlet_importer(mock_import_target, chainlet)
        mock_push.return_value = chain_service

        original_isinstance = self._patch_isinstance_for_mock_service(chain_service)

        try:
            runner = CliRunner()
            chain_path = self._given_test_chain_file()
            result = self._when_invoke_chains_push(runner, chain_path)

            assert result.exit_code == 0, (
                f"Expected exit code 0, got {result.exit_code}. Output: {result.output}"
            )
            self._then_assert_push_called_with_team(mock_push, "team1", "TestChain")
            mock_remote.api.get_chains.assert_called()
        finally:
            self._restore_isinstance(original_isinstance)
