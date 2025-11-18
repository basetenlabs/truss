"""Team resolution logic for training projects."""

from typing import Optional

import click

from truss.cli import remote_cli
from truss.remote.baseten.remote import BasetenRemote


def resolve_training_project_team_id(
    remote_provider: BasetenRemote,
    provided_team_name: Optional[str],
    existing_project_name: Optional[str] = None,
) -> Optional[str]:
    """Resolve team ID from provided team name or by prompting the user.
    This function handles 8 distinct scenarios organized into 3 high-level categories:

    HIGH-LEVEL SCENARIO 1: --team PROVIDED
        SCENARIO 1: Valid team name, user has access
            → Returns the team_id for that team (no prompt, no error)
        SCENARIO 2: Invalid team name (does not exist)
            → Raises ClickException with error message listing available teams

    HIGH-LEVEL SCENARIO 2: --team NOT PROVIDED, Training project does not exist
        SCENARIO 3: User has multiple teams, no existing project
            → Prompts user to select a team via inquire_team()
        SCENARIO 6: User has exactly one team, no existing project
            → Returns the single team_id automatically (no prompt)

    HIGH-LEVEL SCENARIO 3: --team NOT PROVIDED, Training project exists
        SCENARIO 4: User has multiple teams, existing project in exactly one team
            → Auto-detects and returns the team_id for that team (no prompt)
        SCENARIO 5: User has multiple teams, existing project exists in multiple teams
            → Prompts user to select a team via inquire_team()
        SCENARIO 7: User has exactly one team, existing project matches the team
            → Auto-detects and returns the single team_id (no prompt)
        SCENARIO 8: User has exactly one team, existing project exists in different team
            → Returns the single team_id automatically (no prompt, uses user's only team)
    """
    existing_teams = remote_provider.api.get_teams()

    if provided_team_name is not None:
        if provided_team_name not in existing_teams:
            available_teams_str = remote_cli.format_available_teams(existing_teams)
            raise click.ClickException(
                f"Team '{provided_team_name}' does not exist. Available teams: {available_teams_str}"
            )
        return existing_teams[provided_team_name]["id"]

    existing_projects = None
    if existing_project_name is not None:
        existing_projects = remote_provider.api.list_training_projects()
        matching_projects = [
            p for p in existing_projects if p.get("name") == existing_project_name
        ]

        if len(matching_projects) > 1:
            return remote_cli.inquire_team(existing_teams=existing_teams)

        if len(matching_projects) == 1:
            project_team_name = matching_projects[0].get("team_name")
            if project_team_name in existing_teams:
                return existing_teams[project_team_name]["id"]

    if len(existing_teams) == 1:
        single_team = list[dict[str, str]](existing_teams.values())[0]
        return single_team["id"]

    return remote_cli.inquire_team(existing_teams=existing_teams)
