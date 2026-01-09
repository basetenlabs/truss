"""Tests for team parameter in model push and watch.

This test suite covers all 8 scenarios for team resolution in truss push:
1. --team PROVIDED: Valid team name, user has access
2. --team PROVIDED: Invalid team name (does not exist)
3. --team NOT PROVIDED: User has multiple teams, no existing model
4. --team NOT PROVIDED: User has multiple teams, existing model in exactly one team
5. --team NOT PROVIDED: User has multiple teams, existing model exists in multiple teams
6. --team NOT PROVIDED: User has exactly one team, no existing model
7. --team NOT PROVIDED: User has exactly one team, existing model matches the team
8. --team NOT PROVIDED: User has exactly one team, existing model exists in different team

And scenarios for resolve_model_for_watch:
- Single model found: returns it directly
- Multiple models with same name: prompts user to select team
- No model found: raises error
"""

from unittest.mock import Mock, patch

import click
import pytest

from truss.cli.resolvers.model_team_resolver import (
    resolve_model_for_watch,
    resolve_model_team_name,
)
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
        "scenario_num,teams,models_response,existing_model_name,inquire_return,expected_team_name,expected_team_id,should_prompt",
        [
            # SCENARIO 3: Multiple teams, no existing model
            (
                3,
                {
                    "Team Alpha": {"id": "team1", "name": "Team Alpha"},
                    "Team Beta": {"id": "team2", "name": "Team Beta"},
                    "Team Gamma": {"id": "team3", "name": "Team Gamma"},
                },
                {"models": []},  # No models exist
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
                {
                    "models": [
                        {
                            "id": "model1",
                            "name": "existing-model",
                            "team": {"id": "team2", "name": "Team Beta"},
                        }
                    ]
                },
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
                {
                    "models": [
                        {
                            "id": "model1",
                            "name": "existing-model",
                            "team": {"id": "team1", "name": "Team Alpha"},
                        },
                        {
                            "id": "model2",
                            "name": "existing-model",
                            "team": {"id": "team2", "name": "Team Beta"},
                        },
                    ]
                },
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
                {"models": []},  # No models exist
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
                {
                    "models": [
                        {
                            "id": "model1",
                            "name": "existing-model",
                            "team": {"id": "team1", "name": "Team Alpha"},
                        }
                    ]
                },
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
                {
                    "models": [
                        {
                            "id": "model1",
                            "name": "existing-model",
                            "team": {"id": "team2", "name": "Team Other"},
                        }
                    ]
                },
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
        models_response,
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

        mock_remote.api.models.return_value = models_response

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
        if existing_model_name:
            mock_remote.api.models.assert_called_once()

    @pytest.mark.parametrize(
        "existing_teams_param,should_call_get_teams",
        [(None, True), ({"Team Alpha": {"id": "team1", "name": "Team Alpha"}}, False)],
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
        "existing_model_name,should_call_models_api",
        [(None, False), ("some-model", True)],
    )
    @patch("truss.cli.resolvers.model_team_resolver.remote_cli.inquire_team")
    def test_existing_model_name_scenarios(
        self, mock_inquire_team, existing_model_name, should_call_models_api
    ):
        """Test behavior with different existing_model_name values."""
        teams = {
            "Team Alpha": {"id": "team1", "name": "Team Alpha"},
            "Team Beta": {"id": "team2", "name": "Team Beta"},
        }
        mock_remote = self._setup_mock_remote(teams)
        mock_inquire_team.return_value = "Team Beta"
        mock_remote.api.models.return_value = {"models": []}

        team_name, team_id = resolve_model_team_name(
            remote_provider=mock_remote,
            provided_team_name=None,
            existing_model_name=existing_model_name,
            existing_teams=teams,
        )

        assert team_name == "Team Beta"
        assert team_id == "team2"
        mock_inquire_team.assert_called_once_with(existing_teams=teams)
        if should_call_models_api:
            mock_remote.api.models.assert_called_once()
        else:
            mock_remote.api.models.assert_not_called()


class TestResolveModelForWatch:
    """Test model resolution for watch operations with team disambiguation."""

    @staticmethod
    def _setup_mock_remote(teams, models):
        mock_remote = Mock(spec=BasetenRemote)
        mock_api = Mock()
        mock_remote.api = mock_api
        mock_api.get_teams.return_value = teams
        mock_api.get_models_for_watch.return_value = {"models": models}
        return mock_remote

    def test_single_model_found(self):
        """Test that single model is returned directly without prompting."""
        teams = {"Team Alpha": {"id": "team1", "name": "Team Alpha"}}
        models = [
            {
                "name": "my-model",
                "id": "model1",
                "hostname": "host.baseten.co",
                "team": {"id": "team1", "name": "Team Alpha"},
                "versions": [{"id": "v1", "is_draft": True, "is_primary": False}],
            }
        ]
        mock_remote = self._setup_mock_remote(teams, models)

        model, versions = resolve_model_for_watch(mock_remote, "my-model")

        assert model["id"] == "model1"
        assert model["hostname"] == "host.baseten.co"
        assert len(versions) == 1
        assert versions[0]["id"] == "v1"

    def test_no_model_found(self):
        """Test that error is raised when no model is found."""
        teams = {"Team Alpha": {"id": "team1", "name": "Team Alpha"}}
        models = []
        mock_remote = self._setup_mock_remote(teams, models)

        with pytest.raises(click.ClickException) as exc_info:
            resolve_model_for_watch(mock_remote, "nonexistent-model")

        assert "not found" in str(exc_info.value)

    @patch("truss.cli.resolvers.model_team_resolver.remote_cli.inquire_team")
    def test_multiple_models_prompts_for_team(self, mock_inquire_team):
        """Test that user is prompted when multiple models have the same name."""
        teams = {
            "Team Alpha": {"id": "team1", "name": "Team Alpha"},
            "Team Beta": {"id": "team2", "name": "Team Beta"},
        }
        models = [
            {
                "name": "shared-model",
                "id": "model1",
                "hostname": "host1.baseten.co",
                "team": {"id": "team1", "name": "Team Alpha"},
                "versions": [{"id": "v1", "is_draft": True}],
            },
            {
                "name": "shared-model",
                "id": "model2",
                "hostname": "host2.baseten.co",
                "team": {"id": "team2", "name": "Team Beta"},
                "versions": [{"id": "v2", "is_draft": True}],
            },
        ]
        mock_remote = self._setup_mock_remote(teams, models)
        mock_inquire_team.return_value = "Team Beta"

        model, versions = resolve_model_for_watch(mock_remote, "shared-model")

        assert model["id"] == "model2"
        assert model["hostname"] == "host2.baseten.co"
        mock_inquire_team.assert_called_once()
        # Verify that only teams with this model are in the prompt
        call_args = mock_inquire_team.call_args
        teams_in_prompt = call_args[1]["existing_teams"]
        assert "Team Alpha" in teams_in_prompt
        assert "Team Beta" in teams_in_prompt

    def test_model_in_inaccessible_team(self):
        """Test that models in inaccessible teams are filtered out."""
        teams = {"Team Alpha": {"id": "team1", "name": "Team Alpha"}}
        models = [
            {
                "name": "other-model",
                "id": "model1",
                "hostname": "host.baseten.co",
                "team": {"id": "team2", "name": "Team Beta"},  # Not in user's teams
                "versions": [{"id": "v1", "is_draft": True}],
            }
        ]
        mock_remote = self._setup_mock_remote(teams, models)

        with pytest.raises(click.ClickException) as exc_info:
            resolve_model_for_watch(mock_remote, "other-model")

        assert "not found" in str(exc_info.value)

    def test_provided_team_name_valid(self):
        """Test that providing a valid team name filters to that team's model."""
        teams = {
            "Team Alpha": {"id": "team1", "name": "Team Alpha"},
            "Team Beta": {"id": "team2", "name": "Team Beta"},
        }
        models_team1 = [
            {
                "name": "my-model",
                "id": "model1",
                "hostname": "host1.baseten.co",
                "team": {"id": "team1", "name": "Team Alpha"},
                "versions": [{"id": "v1", "is_draft": True}],
            }
        ]
        mock_remote = Mock(spec=BasetenRemote)
        mock_api = Mock()
        mock_remote.api = mock_api
        mock_api.get_teams.return_value = teams
        mock_api.get_models_for_watch.return_value = {"models": models_team1}

        model, versions = resolve_model_for_watch(
            mock_remote, "my-model", provided_team_name="Team Alpha"
        )

        assert model["id"] == "model1"
        # Verify get_models_for_watch was called with team_id
        mock_api.get_models_for_watch.assert_called_once_with(team_id="team1")

    def test_provided_team_name_invalid(self):
        """Test that providing an invalid team name raises an error."""
        teams = {"Team Alpha": {"id": "team1", "name": "Team Alpha"}}
        mock_remote = Mock(spec=BasetenRemote)
        mock_api = Mock()
        mock_remote.api = mock_api
        mock_api.get_teams.return_value = teams

        with pytest.raises(click.ClickException) as exc_info:
            resolve_model_for_watch(
                mock_remote, "my-model", provided_team_name="NonExistent"
            )

        assert "does not exist" in str(exc_info.value)

    def test_provided_team_name_model_not_in_team(self):
        """Test that error is raised when model doesn't exist in provided team."""
        teams = {
            "Team Alpha": {"id": "team1", "name": "Team Alpha"},
            "Team Beta": {"id": "team2", "name": "Team Beta"},
        }
        # Model exists in team2, but we're querying team1
        models_team1 = []  # No models in team1
        mock_remote = Mock(spec=BasetenRemote)
        mock_api = Mock()
        mock_remote.api = mock_api
        mock_api.get_teams.return_value = teams
        mock_api.get_models_for_watch.return_value = {"models": models_team1}

        with pytest.raises(click.ClickException) as exc_info:
            resolve_model_for_watch(
                mock_remote, "my-model", provided_team_name="Team Alpha"
            )

        assert "not found in team" in str(exc_info.value)
