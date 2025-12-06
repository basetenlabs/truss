"""Team resolution logic for models."""

from typing import Optional

import click

from truss.cli import remote_cli
from truss.remote.baseten.remote import BasetenRemote


def resolve_model_team_name(
    remote_provider: BasetenRemote,
    provided_team_name: Optional[str],
    existing_model_name: Optional[str] = None,
    existing_teams: Optional[dict[str, dict[str, str]]] = None,
) -> tuple[Optional[str], Optional[str]]:
    """Resolve team name and team_id from provided team name or by prompting the user.
    Returns a tuple of (team_name, team_id).
    This function handles 8 distinct scenarios organized into 3 high-level categories:

    HIGH-LEVEL SCENARIO 1: --team PROVIDED
        SCENARIO 1: Valid team name, user has access
            → Returns (team_name, team_id) for that team (no prompt, no error)
        SCENARIO 2: Invalid team name (does not exist)
            → Raises ClickException with error message listing available teams

    HIGH-LEVEL SCENARIO 2: --team NOT PROVIDED, Model does not exist
        SCENARIO 3: User has multiple teams, no existing model
            → Prompts user to select a team via inquire_team()
        SCENARIO 6: User has exactly one team, no existing model
            → Returns (team_name, team_id) for the single team automatically (no prompt)

    HIGH-LEVEL SCENARIO 3: --team NOT PROVIDED, Model exists
        SCENARIO 4: User has multiple teams, existing model in exactly one team
            → Auto-detects and returns (team_name, team_id) for that team (no prompt)
        SCENARIO 5: User has multiple teams, existing model exists in multiple teams
            → Prompts user to select a team via inquire_team()
        SCENARIO 7: User has exactly one team, existing model matches the team
            → Auto-detects and returns (team_name, team_id) for the single team (no prompt)
        SCENARIO 8: User has exactly one team, existing model exists in different team
            → Returns (team_name, team_id) for the single team automatically (no prompt, uses user's only team)
    """
    if existing_teams is None:
        existing_teams = remote_provider.api.get_teams()

    def _get_team_id(team_name: Optional[str]) -> Optional[str]:
        if team_name and existing_teams:
            team_data = existing_teams.get(team_name)
            return team_data["id"] if team_data else None
        return None

    def _get_matching_models_in_accessible_teams(model_name: str) -> list[dict]:
        """Get models matching the name that are in teams the user has access to."""
        all_models_data = remote_provider.api.models()
        accessible_team_ids = {team_data["id"] for team_data in existing_teams.values()}

        return [
            m
            for m in all_models_data.get("models", [])
            if m.get("name") == model_name
            and m.get("team", {}).get("id") in accessible_team_ids
        ]

    if provided_team_name is not None:
        if provided_team_name not in existing_teams:
            available_teams_str = remote_cli.format_available_teams(existing_teams)
            raise click.ClickException(
                f"Team '{provided_team_name}' does not exist. Available teams: {available_teams_str}"
            )
        return (provided_team_name, _get_team_id(provided_team_name))

    if existing_model_name is not None:
        matching_models = _get_matching_models_in_accessible_teams(existing_model_name)

        if len(matching_models) == 1:
            # Exactly one model in an accessible team - auto-detect
            team = matching_models[0].get("team", {})
            model_team_name = team.get("name")
            model_team_id = team.get("id")
            if model_team_name and model_team_name in existing_teams:
                return (model_team_name, model_team_id)
        # If len > 1, multiple models exist - fall through to prompt logic
        # If len == 0, no models exist - fall through to prompt logic

    if len(existing_teams) == 1:
        single_team_name = list(existing_teams.keys())[0]
        return (single_team_name, _get_team_id(single_team_name))

    selected_team_name = remote_cli.inquire_team(existing_teams=existing_teams)
    return (selected_team_name, _get_team_id(selected_team_name))
