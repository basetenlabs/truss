"""Tests for team parameter in model push.

This test suite covers all 8 scenarios for team resolution in truss push:
1. --team PROVIDED: Valid team name, user has access
2. --team PROVIDED: Invalid team name (does not exist)
3. --team NOT PROVIDED: User has multiple teams, no existing model
4. --team NOT PROVIDED: User has multiple teams, existing model in exactly one team
5. --team NOT PROVIDED: User has multiple teams, existing model exists in multiple teams
6. --team NOT PROVIDED: User has exactly one team, no existing model
7. --team NOT PROVIDED: User has exactly one team, existing model matches the team
8. --team NOT PROVIDED: User has exactly one team, existing model exists in different team
"""

from unittest.mock import Mock, patch

import click
import pytest

from truss.cli.resolvers.model_team_resolver import resolve_model_team_name
from truss.remote.baseten.remote import BasetenRemote


class TestModelTeamResolver:
    """Test team parameter resolution for model push."""

    @staticmethod
    def _setup_mock_remote(teams):
        mock_remote = Mock(spec=BasetenRemote)
        mock_api = Mock()
        mock_remote.api = mock_api
        mock_api.get_teams.return_value = teams
        return mock_remote

    @pytest.mark.parametrize(
        "provided_team_name,expected_team_name,expected_team_id,should_raise",
        [
            # SCENARIO 1: Valid team name
            ("Team Alpha", "Team Alpha", "team1", False),
            # SCENARIO 2: Invalid team name
            ("NonExistentTeam", None, None, True),
        ],
    )
    def test_team_provided_scenarios(
        self, provided_team_name, expected_team_name, expected_team_id, should_raise
    ):
        """Test scenarios when --team is provided."""
        teams = {"Team Alpha": {"id": "team1", "name": "Team Alpha"}}
        mock_remote = self._setup_mock_remote(teams)

        if should_raise:
            with pytest.raises(click.ClickException) as exc_info:
                resolve_model_team_name(
                    remote_provider=mock_remote,
                    provided_team_name=provided_team_name,
                    existing_teams=teams,
                )
            assert "does not exist" in str(exc_info.value)
            assert provided_team_name in str(exc_info.value)
        else:
            team_name, team_id = resolve_model_team_name(
                remote_provider=mock_remote,
                provided_team_name=provided_team_name,
                existing_teams=teams,
            )
            assert team_name == expected_team_name
            assert team_id == expected_team_id
            mock_remote.api.get_teams.assert_not_called()

    @pytest.mark.parametrize(
        "scenario_num,teams,model_team_id,existing_model_name,inquire_return,expected_team_name,expected_team_id,should_prompt",
        [
            # SCENARIO 3: Multiple teams, no existing model
            (
                3,
                {
                    "Team Alpha": {"id": "team1", "name": "Team Alpha"},
                    "Team Beta": {"id": "team2", "name": "Team Beta"},
                    "Team Gamma": {"id": "team3", "name": "Team Gamma"},
                },
                None,
                "non-existent-model",
                "Team Beta",
                "Team Beta",
                "team2",
                True,
            ),
            # SCENARIO 4: Multiple teams, existing model in exactly one team
            (
                4,
                {
                    "Team Alpha": {"id": "team1", "name": "Team Alpha"},
                    "Team Beta": {"id": "team2", "name": "Team Beta"},
                    "Team Gamma": {"id": "team3", "name": "Team Gamma"},
                },
                "team2",
                "existing-model",
                None,
                "Team Beta",
                "team2",
                False,
            ),
            # SCENARIO 5: Multiple teams, existing model in multiple teams
            (
                5,
                {
                    "Team Alpha": {"id": "team1", "name": "Team Alpha"},
                    "Team Beta": {"id": "team2", "name": "Team Beta"},
                    "Team Gamma": {"id": "team3", "name": "Team Gamma"},
                },
                None,
                "existing-model",
                "Team Alpha",
                "Team Alpha",
                "team1",
                True,
            ),
            # SCENARIO 6: Single team, no existing model
            (
                6,
                {"Team Alpha": {"id": "team1", "name": "Team Alpha"}},
                None,
                "non-existent-model",
                None,
                "Team Alpha",
                "team1",
                False,
            ),
            # SCENARIO 7: Single team, existing model matches the team
            (
                7,
                {"Team Alpha": {"id": "team1", "name": "Team Alpha"}},
                "team1",
                "existing-model",
                None,
                "Team Alpha",
                "team1",
                False,
            ),
            # SCENARIO 8: Single team, existing model in different team
            (
                8,
                {"Team Alpha": {"id": "team1", "name": "Team Alpha"}},
                "team2",
                "existing-model",
                None,
                "Team Alpha",
                "team1",
                False,
            ),
        ],
    )
    @patch("truss.cli.resolvers.model_team_resolver.remote_cli.inquire_team")
    def test_team_not_provided_scenarios(
        self,
        mock_inquire_team,
        scenario_num,
        teams,
        model_team_id,
        existing_model_name,
        inquire_return,
        expected_team_name,
        expected_team_id,
        should_prompt,
    ):
        """Test scenarios when --team is NOT provided."""
        mock_remote = self._setup_mock_remote(teams)
        if inquire_return:
            mock_inquire_team.return_value = inquire_return

        with patch(
            "truss.cli.resolvers.model_team_resolver.get_model_team_id",
            return_value=model_team_id,
        ):
            team_name, team_id = resolve_model_team_name(
                remote_provider=mock_remote,
                provided_team_name=None,
                existing_model_name=existing_model_name,
                existing_teams=teams,
            )

        assert team_name == expected_team_name
        assert team_id == expected_team_id
        if should_prompt:
            mock_inquire_team.assert_called_once_with(existing_teams=teams)
        else:
            mock_inquire_team.assert_not_called()

    @pytest.mark.parametrize(
        "existing_teams_param,should_call_get_teams",
        [
            (None, True),
            ({"Team Alpha": {"id": "team1", "name": "Team Alpha"}}, False),
        ],
    )
    def test_get_teams_called_when_existing_teams_none(
        self, existing_teams_param, should_call_get_teams
    ):
        """Test that get_teams is called when existing_teams is not provided."""
        teams = {"Team Alpha": {"id": "team1", "name": "Team Alpha"}}
        mock_remote = self._setup_mock_remote(teams)

        team_name, team_id = resolve_model_team_name(
            remote_provider=mock_remote,
            provided_team_name="Team Alpha",
            existing_teams=existing_teams_param,
        )

        assert team_name == "Team Alpha"
        assert team_id == "team1"
        if should_call_get_teams:
            mock_remote.api.get_teams.assert_called_once()
        else:
            mock_remote.api.get_teams.assert_not_called()

    @pytest.mark.parametrize(
        "existing_model_name,should_call_get_model_team_id",
        [
            (None, False),
            ("some-model", True),
        ],
    )
    @patch("truss.cli.resolvers.model_team_resolver.remote_cli.inquire_team")
    def test_existing_model_name_scenarios(
        self, mock_inquire_team, existing_model_name, should_call_get_model_team_id
    ):
        """Test behavior with different existing_model_name values."""
        teams = {
            "Team Alpha": {"id": "team1", "name": "Team Alpha"},
            "Team Beta": {"id": "team2", "name": "Team Beta"},
        }
        mock_remote = self._setup_mock_remote(teams)
        mock_inquire_team.return_value = "Team Beta"

        with patch(
            "truss.cli.resolvers.model_team_resolver.get_model_team_id", return_value=None
        ) as mock_get_model_team_id:
            team_name, team_id = resolve_model_team_name(
                remote_provider=mock_remote,
                provided_team_name=None,
                existing_model_name=existing_model_name,
                existing_teams=teams,
            )

        assert team_name == "Team Beta"
        assert team_id == "team2"
        mock_inquire_team.assert_called_once_with(existing_teams=teams)
        if should_call_get_model_team_id:
            mock_get_model_team_id.assert_called_once_with(
                mock_remote.api, existing_model_name
            )
        else:
            mock_get_model_team_id.assert_not_called()

