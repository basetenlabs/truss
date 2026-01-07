"""Team resolution logic for models."""

from typing import Callable, Optional

import click

from truss.cli import remote_cli
from truss.remote.baseten.remote import BasetenRemote


def _validate_provided_team(
    provided_team_name: str, existing_teams: dict[str, dict[str, str]]
) -> str:
    """Validate provided team name exists and return team_id."""
    if provided_team_name not in existing_teams:
        available_teams_str = remote_cli.format_available_teams(existing_teams)
        raise click.ClickException(
            f"Team '{provided_team_name}' does not exist. Available teams: {available_teams_str}"
        )
    return existing_teams[provided_team_name]["id"]


def _get_matching_models(
    model_name: str,
    existing_teams: dict[str, dict[str, str]],
    fetch_models: Callable[[], dict],
) -> list[dict]:
    """Get models matching name that are in accessible teams."""
    all_models_data = fetch_models()
    accessible_team_ids = {team_data["id"] for team_data in existing_teams.values()}
    return [
        m
        for m in all_models_data.get("models", [])
        if m.get("name") == model_name
        and m.get("team", {}).get("id") in accessible_team_ids
    ]


def _prompt_for_team_from_models(
    matching_models: list[dict], existing_teams: dict[str, dict[str, str]], prompt: str
) -> dict:
    """Prompt user to select team when multiple models match, return selected model."""
    team_name_to_model = {
        m.get("team", {}).get("name", "Unknown"): m for m in matching_models
    }
    teams_with_model = {
        name: existing_teams[name]
        for name in team_name_to_model
        if name in existing_teams
    }
    if not teams_with_model:
        raise click.ClickException(
            "Model exists but you don't have access to any team that owns it."
        )
    selected = remote_cli.inquire_team(existing_teams=teams_with_model, prompt=prompt)
    if not selected or selected not in team_name_to_model:
        raise click.ClickException("No team selected.")
    return team_name_to_model[selected]


def resolve_model_for_watch(
    remote_provider: BasetenRemote,
    model_name: str,
    provided_team_name: Optional[str] = None,
    prompt: str = "👥 Multiple models with this name exist. Which team's model do you want to watch?",
) -> tuple[dict, list]:
    """Resolve a model by name for watch, handling team disambiguation.

    Returns tuple of (model_dict, versions_list).
    """
    existing_teams = remote_provider.api.get_teams()

    if provided_team_name is not None:
        team_id = _validate_provided_team(provided_team_name, existing_teams)
        models_data = remote_provider.api.get_models_for_watch(team_id=team_id)
        matching = [
            m for m in models_data.get("models", []) if m.get("name") == model_name
        ]
        if not matching:
            raise click.ClickException(
                f"Model '{model_name}' not found in team '{provided_team_name}'."
            )
        return matching[0], matching[0].get("versions", [])

    matching_models = _get_matching_models(
        model_name, existing_teams, remote_provider.api.get_models_for_watch
    )

    if not matching_models:
        raise click.ClickException(f"Model '{model_name}' not found.")

    if len(matching_models) == 1:
        return matching_models[0], matching_models[0].get("versions", [])

    model = _prompt_for_team_from_models(matching_models, existing_teams, prompt)
    return model, model.get("versions", [])


def resolve_model_team_name(
    remote_provider: BasetenRemote,
    provided_team_name: Optional[str],
    existing_model_name: Optional[str] = None,
    existing_teams: Optional[dict[str, dict[str, str]]] = None,
) -> tuple[Optional[str], Optional[str]]:
    """Resolve team name and team_id for push operations.

    Returns tuple of (team_name, team_id).
    """
    if existing_teams is None:
        existing_teams = remote_provider.api.get_teams()

    def _get_team_id(team_name: Optional[str]) -> Optional[str]:
        if team_name and existing_teams:
            team_data = existing_teams.get(team_name)
            return team_data["id"] if team_data else None
        return None

    if provided_team_name is not None:
        _validate_provided_team(provided_team_name, existing_teams)
        return (provided_team_name, _get_team_id(provided_team_name))

    if existing_model_name is not None:
        matching_models = _get_matching_models(
            existing_model_name, existing_teams, remote_provider.api.models
        )
        if len(matching_models) == 1:
            team = matching_models[0].get("team", {})
            model_team_name = team.get("name")
            model_team_id = team.get("id")
            if model_team_name and model_team_name in existing_teams:
                return (model_team_name, model_team_id)

    if len(existing_teams) == 1:
        single_team_name = list(existing_teams.keys())[0]
        return (single_team_name, _get_team_id(single_team_name))

    selected_team_name = remote_cli.inquire_team(existing_teams=existing_teams)
    return (selected_team_name, _get_team_id(selected_team_name))
