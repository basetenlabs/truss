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

    This function handles three scenarios:
    1. Training project does not exist → create in specified team (or default/prompt)
    2. Training project exists in 1 team → eagerly assign to that team
    3. Training project exists in multiple teams → user must specify --team or prompt

    Args:
        remote_provider: The remote provider to fetch teams from
        provided_team_name: Optional team name provided via CLI
        existing_project_name: Optional training project name to check for existing project

    Returns:
        Team ID if resolved, None otherwise

    Raises:
        click.ClickException: If provided team name doesn't exist
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
            for team_name, team_info in existing_teams.items():
                if team_name == project_team_name:
                    return team_info["id"]
            return remote_cli.inquire_team(existing_teams=existing_teams)

    if len(existing_teams) == 1:
        single_team = list(existing_teams.values())[0]
        single_team_id = single_team["id"]
        single_team_name = single_team["name"]

        if existing_project_name is not None and existing_projects is not None:
            for project in existing_projects:
                if project.get("name") == existing_project_name:
                    project_team_name = project.get("team_name")
                    if project_team_name == single_team_name:
                        return single_team_id
                    break

        return single_team_id

    return remote_cli.inquire_team(existing_teams=existing_teams)
