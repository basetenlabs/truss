"""Team resolution logic for chains."""

from typing import Callable, Optional

import click

from truss.cli import remote_cli
from truss.remote.baseten.custom_types import TeamType
from truss.remote.baseten.remote import BasetenRemote


def _validate_provided_team(
    provided_team_name: str, existing_teams: dict[str, TeamType]
) -> str:
    """Validate provided team name exists and return team_id."""
    if provided_team_name not in existing_teams:
        available_teams_str = remote_cli.format_available_teams(existing_teams)
        raise click.ClickException(
            f"Team '{provided_team_name}' does not exist. Available teams: {available_teams_str}"
        )
    return existing_teams[provided_team_name].id


def _get_matching_chains(
    chain_name: str,
    existing_teams: dict[str, TeamType],
    fetch_chains: Callable[[], list],
) -> list[dict]:
    """Get chains matching name that are in accessible teams."""
    all_chains = fetch_chains()
    accessible_team_ids = {team_data.id for team_data in existing_teams.values()}
    return [
        c
        for c in all_chains
        if c.get("name") == chain_name
        and c.get("team", {}).get("id") in accessible_team_ids
    ]


def _prompt_for_team_from_chains(
    matching_chains: list[dict], existing_teams: dict[str, TeamType], prompt: str
) -> dict:
    """Prompt user to select team when multiple chains match, return selected chain."""
    team_name_to_chain = {
        c.get("team", {}).get("name", "Unknown"): c for c in matching_chains
    }
    teams_with_chain = {
        name: existing_teams[name]
        for name in team_name_to_chain
        if name in existing_teams
    }
    if not teams_with_chain:
        raise click.ClickException(
            "Chain exists but you don't have access to any team that owns it."
        )
    selected = remote_cli.inquire_team(existing_teams=teams_with_chain, prompt=prompt)
    if not selected or selected not in team_name_to_chain:
        raise click.ClickException("No team selected.")
    return team_name_to_chain[selected]


def resolve_chain_for_watch(
    remote_provider: BasetenRemote,
    chain_name: str,
    provided_team_name: Optional[str] = None,
    prompt: str = "Multiple chains with this name exist. Which team's chain do you want to watch?",
) -> dict:
    """Resolve a chain by name for watch, handling team disambiguation.

    Returns chain dict with 'id', 'name', and 'team' info.
    """
    existing_teams = remote_provider.api.get_teams()

    if provided_team_name is not None:
        team_id = _validate_provided_team(provided_team_name, existing_teams)
        chains = remote_provider.api.get_chains(team_id=team_id)
        matching = [c for c in chains if c.get("name") == chain_name]
        if not matching:
            raise click.ClickException(
                f"Chain '{chain_name}' not found in team '{provided_team_name}'."
            )
        return matching[0]

    matching_chains = _get_matching_chains(
        chain_name, existing_teams, remote_provider.api.get_chains
    )

    if not matching_chains:
        raise click.ClickException(f"Chain '{chain_name}' not found.")

    if len(matching_chains) == 1:
        return matching_chains[0]

    return _prompt_for_team_from_chains(matching_chains, existing_teams, prompt)


def resolve_chain_team_name(
    remote_provider: BasetenRemote,
    provided_team_name: Optional[str],
    existing_chain_name: Optional[str] = None,
    existing_teams: Optional[dict[str, TeamType]] = None,
) -> tuple[Optional[str], Optional[str]]:
    """Resolve team name and team_id from provided team name or by prompting the user.
    Returns a tuple of (team_name, team_id).
    This function handles 8 distinct scenarios organized into 3 high-level categories:

    HIGH-LEVEL SCENARIO 1: --team PROVIDED
        SCENARIO 1: Valid team name, user has access
            → Returns (team_name, team_id) for that team (no prompt, no error)
        SCENARIO 2: Invalid team name (does not exist)
            → Raises ClickException with error message listing available teams

    HIGH-LEVEL SCENARIO 2: --team NOT PROVIDED, Chain does not exist
        SCENARIO 3: User has multiple teams, no existing chain
            → Prompts user to select a team via inquire_team()
        SCENARIO 6: User has exactly one team, no existing chain
            → Returns (team_name, team_id) for the single team automatically (no prompt)

    HIGH-LEVEL SCENARIO 3: --team NOT PROVIDED, Chain exists
        SCENARIO 4: User has multiple teams, existing chain in exactly one team
            → Auto-detects and returns (team_name, team_id) for that team (no prompt)
        SCENARIO 5: User has multiple teams, existing chain exists in multiple teams
            → Prompts user to select a team via inquire_team()
        SCENARIO 7: User has exactly one team, existing chain matches the team
            → Auto-detects and returns (team_name, team_id) for the single team (no prompt)
        SCENARIO 8: User has exactly one team, existing chain exists in different team
            → Returns (team_name, team_id) for the single team automatically (no prompt, uses user's only team)
    """
    if existing_teams is None:
        existing_teams = remote_provider.api.get_teams()

    def _get_team_id(team_name: Optional[str]) -> Optional[str]:
        if team_name and existing_teams:
            team_data = existing_teams.get(team_name)
            return team_data.id if team_data else None
        return None

    if provided_team_name is not None:
        if provided_team_name not in existing_teams:
            available_teams_str = remote_cli.format_available_teams(existing_teams)
            raise click.ClickException(
                f"Team '{provided_team_name}' does not exist. Available teams: {available_teams_str}"
            )
        return (provided_team_name, _get_team_id(provided_team_name))

    existing_chains = None
    if existing_chain_name is not None:
        existing_chains = remote_provider.api.get_chains()
        matching_chains = [
            c for c in existing_chains if c.get("name") == existing_chain_name
        ]

        if len(matching_chains) > 1:
            selected_team_name = remote_cli.inquire_team(existing_teams=existing_teams)
            return (selected_team_name, _get_team_id(selected_team_name))

        if len(matching_chains) == 1:
            chain_team = matching_chains[0].get("team")
            chain_team_name = chain_team.get("name") if chain_team else None
            if chain_team_name and chain_team_name in existing_teams:
                return (chain_team_name, _get_team_id(chain_team_name))

    if len(existing_teams) == 1:
        single_team_name = list(existing_teams.keys())[0]
        return (single_team_name, _get_team_id(single_team_name))

    selected_team_name = remote_cli.inquire_team(existing_teams=existing_teams)
    return (selected_team_name, _get_team_id(selected_team_name))
