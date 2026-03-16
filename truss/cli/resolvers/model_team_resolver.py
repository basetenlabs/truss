"""Team resolution logic for models."""

from typing import Callable, Optional

import click

from truss.cli import remote_cli
from truss.remote.baseten.custom_types import TeamType
from truss.remote.baseten.remote import BasetenRemote
from truss.remote.remote_factory import RemoteFactory


def _get_matching_models(
    model_name: str,
    existing_teams: dict[str, TeamType],
    fetch_models: Callable[[], dict],
) -> list[dict]:
    """Get models matching name that are in accessible teams."""
    all_models_data = fetch_models()
    accessible_team_ids = {team_data.id for team_data in existing_teams.values()}
    return [
        m
        for m in all_models_data.get("models", [])
        if m.get("name") == model_name
        and m.get("team", {}).get("id") in accessible_team_ids
    ]


def _prompt_for_team_from_models(
    matching_models: list[dict], existing_teams: dict[str, TeamType], prompt: str
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
    chainlets_only: bool = False,
) -> tuple[dict, list]:
    """Resolve a model by name for watch, handling team disambiguation.

    Args:
        remote_provider: The Baseten remote provider.
        model_name: The name of the model to resolve.
        provided_team_name: Optional team name to filter by.
        prompt: Prompt message for team selection.
        chainlets_only: If True, query chainlet oracles (origin=CHAINS) instead of
            regular models (origin=BASETEN). Required for chains watch.

    Returns tuple of (model_dict, versions_list).
    """
    existing_teams = remote_provider.api.get_teams()

    if provided_team_name is not None:
        available_teams_str = remote_cli.format_available_teams(existing_teams)
        if provided_team_name not in existing_teams:
            raise click.ClickException(
                f"Team '{provided_team_name}' does not exist. Available teams: {available_teams_str}"
            )
        team_id = existing_teams[provided_team_name].id
        models_data = remote_provider.api.get_models_for_watch(
            team_id=team_id, chainlets_only=chainlets_only
        )
        matching = [
            m for m in models_data.get("models", []) if m.get("name") == model_name
        ]
        if not matching:
            raise click.ClickException(
                f"Model '{model_name}' not found in team '{provided_team_name}'."
            )
        return matching[0], matching[0].get("versions", [])

    matching_models = _get_matching_models(
        model_name,
        existing_teams,
        lambda: remote_provider.api.get_models_for_watch(chainlets_only=chainlets_only),
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
    existing_teams: Optional[dict[str, TeamType]] = None,
    allow_interactive: bool = True,
    remote_name: Optional[str] = None,
) -> tuple[Optional[str], Optional[str]]:
    """Resolve team name and team_id for a model push.

    Returns a tuple of (team_name, team_id).

    Raises ValueError when resolution fails (invalid team name, or ambiguous
    teams when allow_interactive=False).

    Resolution order:
        1. Use provided_team_name if given, or fall back to .trussrc config
           via remote_name.
        2. If model already exists in exactly one accessible team, use that team.
        3. If there is exactly one team, use it.
        4. If allow_interactive, prompt the user. Otherwise raise ValueError.

    Args:
        remote_provider: The Baseten remote provider.
        provided_team_name: Optional team name provided by the user.
        existing_model_name: Optional model name to check for existing models.
        existing_teams: Optional pre-fetched teams dict. If None, will be fetched.
        allow_interactive: If True (default), prompt the user when team is ambiguous.
            If False, raise ValueError instead.
        remote_name: Optional remote name to look up team from .trussrc config as
            a fallback when provided_team_name is None.
    """
    if existing_teams is None:
        existing_teams = remote_provider.api.get_teams()

    def _get_team_id(team_name: Optional[str]) -> Optional[str]:
        if team_name and existing_teams:
            team_data = existing_teams.get(team_name)
            return team_data.id if team_data else None
        return None

    # Fall back to .trussrc config if no team name was explicitly provided
    effective_team_name = provided_team_name
    if effective_team_name is None and remote_name is not None:
        effective_team_name = RemoteFactory.get_remote_team(remote_name)

    if effective_team_name is not None:
        if effective_team_name not in existing_teams:
            available_teams_str = remote_cli.format_available_teams(existing_teams)
            raise ValueError(
                f"Team '{effective_team_name}' does not exist. "
                f"Available teams: {available_teams_str}"
            )
        return (effective_team_name, _get_team_id(effective_team_name))

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

    if not allow_interactive:
        available_teams_str = remote_cli.format_available_teams(existing_teams)
        raise ValueError(
            "Multiple teams available. Please specify a team name via the "
            f"`team` parameter. Available teams: {available_teams_str}"
        )

    selected_team_name = remote_cli.inquire_team(existing_teams=existing_teams)
    return (selected_team_name, _get_team_id(selected_team_name))
