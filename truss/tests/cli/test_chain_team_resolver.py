"""Tests for resolve_chain_for_watch in chain_team_resolver.

This test suite covers scenarios for chain resolution with team disambiguation:
- Single chain found: returns it directly
- Multiple chains with same name: prompts user to select team
- No chain found: raises error
- Provided team name filters to that team
"""

from unittest.mock import Mock, patch

import click
import pytest

from truss.cli.resolvers.chain_team_resolver import resolve_chain_for_watch
from truss.remote.baseten.custom_types import TeamType
from truss.remote.baseten.remote import BasetenRemote


class TestResolveChainForWatch:
    """Test chain resolution for watch operations with team disambiguation."""

    @staticmethod
    def _setup_mock_remote(teams, chains):
        mock_remote = Mock(spec=BasetenRemote)
        mock_api = Mock()
        mock_remote.api = mock_api
        # Convert dictionaries to TeamType objects
        teams_with_type = {
            name: TeamType(**team_data) for name, team_data in teams.items()
        }
        mock_api.get_teams.return_value = teams_with_type
        mock_api.get_chains.return_value = chains
        return mock_remote

    def test_single_chain_found(self):
        """Test that single chain is returned directly without prompting."""
        teams = {"Team Alpha": {"id": "team1", "name": "Team Alpha", "default": True}}
        chains = [
            {
                "name": "my-chain",
                "id": "chain1",
                "team": {"id": "team1", "name": "Team Alpha"},
            }
        ]
        mock_remote = self._setup_mock_remote(teams, chains)

        resolved_chain = resolve_chain_for_watch(mock_remote, "my-chain")

        assert resolved_chain["id"] == "chain1"
        assert resolved_chain["name"] == "my-chain"

    def test_no_chain_found(self):
        """Test that error is raised when no chain is found."""
        teams = {"Team Alpha": {"id": "team1", "name": "Team Alpha", "default": True}}
        chains = []
        mock_remote = self._setup_mock_remote(teams, chains)

        with pytest.raises(click.ClickException) as exc_info:
            resolve_chain_for_watch(mock_remote, "nonexistent-chain")

        assert "not found" in str(exc_info.value)

    @patch("truss.cli.resolvers.chain_team_resolver.remote_cli.inquire_team")
    def test_multiple_chains_prompts_for_team(self, mock_inquire_team):
        """Test that user is prompted when multiple chains have the same name."""
        teams = {
            "Team Alpha": {"id": "team1", "name": "Team Alpha", "default": True},
            "Team Beta": {"id": "team2", "name": "Team Beta", "default": False},
        }
        chains = [
            {
                "name": "shared-chain",
                "id": "chain1",
                "team": {"id": "team1", "name": "Team Alpha"},
            },
            {
                "name": "shared-chain",
                "id": "chain2",
                "team": {"id": "team2", "name": "Team Beta"},
            },
        ]
        mock_remote = self._setup_mock_remote(teams, chains)
        mock_inquire_team.return_value = "Team Beta"

        resolved_chain = resolve_chain_for_watch(mock_remote, "shared-chain")

        assert resolved_chain["id"] == "chain2"
        mock_inquire_team.assert_called_once()
        # Verify that only teams with this chain are in the prompt
        call_args = mock_inquire_team.call_args
        teams_in_prompt = call_args[1]["existing_teams"]
        assert "Team Alpha" in teams_in_prompt
        assert "Team Beta" in teams_in_prompt

    def test_chain_in_inaccessible_team(self):
        """Test that chains in inaccessible teams are filtered out."""
        teams = {"Team Alpha": {"id": "team1", "name": "Team Alpha", "default": True}}
        chains = [
            {
                "name": "other-chain",
                "id": "chain1",
                "team": {"id": "team2", "name": "Team Beta"},  # Not in user's teams
            }
        ]
        mock_remote = self._setup_mock_remote(teams, chains)

        with pytest.raises(click.ClickException) as exc_info:
            resolve_chain_for_watch(mock_remote, "other-chain")

        assert "not found" in str(exc_info.value)

    def test_provided_team_name_valid(self):
        """Test that providing a valid team name filters to that team's chain."""
        teams = {
            "Team Alpha": {"id": "team1", "name": "Team Alpha", "default": True},
            "Team Beta": {"id": "team2", "name": "Team Beta", "default": False},
        }
        chains_team1 = [
            {
                "name": "my-chain",
                "id": "chain1",
                "team": {"id": "team1", "name": "Team Alpha"},
            }
        ]
        mock_remote = Mock(spec=BasetenRemote)
        mock_api = Mock()
        mock_remote.api = mock_api
        # Convert dictionaries to TeamType objects
        teams_with_type = {
            name: TeamType(**team_data) for name, team_data in teams.items()
        }
        mock_api.get_teams.return_value = teams_with_type
        mock_api.get_chains.return_value = chains_team1

        resolved_chain = resolve_chain_for_watch(
            mock_remote, "my-chain", provided_team_name="Team Alpha"
        )

        assert resolved_chain["id"] == "chain1"
        # Verify get_chains was called with team_id
        mock_api.get_chains.assert_called_once_with(team_id="team1")

    def test_provided_team_name_invalid(self):
        """Test that providing an invalid team name raises an error."""
        teams = {"Team Alpha": {"id": "team1", "name": "Team Alpha", "default": True}}
        mock_remote = Mock(spec=BasetenRemote)
        mock_api = Mock()
        mock_remote.api = mock_api
        # Convert dictionaries to TeamType objects
        teams_with_type = {
            name: TeamType(**team_data) for name, team_data in teams.items()
        }
        mock_api.get_teams.return_value = teams_with_type

        with pytest.raises(click.ClickException) as exc_info:
            resolve_chain_for_watch(
                mock_remote, "my-chain", provided_team_name="NonExistent"
            )

        assert "does not exist" in str(exc_info.value)

    def test_provided_team_name_chain_not_in_team(self):
        """Test that error is raised when chain doesn't exist in provided team."""
        teams = {
            "Team Alpha": {"id": "team1", "name": "Team Alpha", "default": True},
            "Team Beta": {"id": "team2", "name": "Team Beta", "default": False},
        }
        # Chain exists in team2, but we're querying team1
        chains_team1 = []  # No chains in team1
        mock_remote = Mock(spec=BasetenRemote)
        mock_api = Mock()
        mock_remote.api = mock_api
        # Convert dictionaries to TeamType objects
        teams_with_type = {
            name: TeamType(**team_data) for name, team_data in teams.items()
        }
        mock_api.get_teams.return_value = teams_with_type
        mock_api.get_chains.return_value = chains_team1

        with pytest.raises(click.ClickException) as exc_info:
            resolve_chain_for_watch(
                mock_remote, "my-chain", provided_team_name="Team Alpha"
            )

        assert "not found in team" in str(exc_info.value)
