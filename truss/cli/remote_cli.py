from typing import Dict, List, Optional

from InquirerPy import inquirer
from InquirerPy.validator import ValidationError, Validator

from truss.cli.utils.output import console
from truss.remote.baseten.remote import BasetenRemote
from truss.remote.remote_factory import USER_TRUSSRC_PATH, RemoteFactory
from truss.remote.truss_remote import RemoteConfig


class NonEmptyValidator(Validator):
    def validate(self, document):
        if not document.text.strip():
            raise ValidationError(
                message="Please enter a non-empty value",
                cursor_position=len(document.text),
            )


def inquire_remote_config() -> RemoteConfig:
    # TODO(bola): extract questions from remote
    console.print("💻 Let's add a Baseten remote!")
    # If users need to adjust the remote url, they
    # can do so manually in the .trussrc file.
    remote_url = "https://app.baseten.co"
    api_key = inquirer.secret(
        message="🤫 Quietly paste your API_KEY:", qmark="", validate=NonEmptyValidator()
    ).execute()
    return RemoteConfig(
        name="baseten",
        configs={
            "remote_provider": "baseten",
            "api_key": api_key,
            "remote_url": remote_url,
        },
    )


def inquire_remote_name() -> str:
    available_remotes = RemoteFactory.get_available_config_names()
    if len(available_remotes) > 1:
        remote = inquirer.select(
            "🎮 Which remote do you want to connect to?",
            qmark="",
            choices=available_remotes,
        ).execute()
        return remote
    elif len(available_remotes) == 1:
        return available_remotes[0]
    remote_config = inquire_remote_config()
    RemoteFactory.update_remote_config(remote_config)

    console.print(
        f"💾 Remote config `{remote_config.name}` saved to `{USER_TRUSSRC_PATH}`."
    )
    return remote_config.name


def inquire_model_name() -> str:
    return inquirer.text("📦 Name this model:", qmark="").execute()


def get_team_id_from_name(teams: Dict[str, Dict[str, str]], team_name: str) -> Optional[str]:
    """
    Get team ID from team name given a dictionary mapping team name to team data.
    Returns team ID if found, None otherwise.
    """
    team = teams.get(team_name)
    return team["id"] if team else None


def format_available_teams(teams: Dict[str, Dict[str, str]]) -> str:
    """
    Format a dictionary of teams into a comma-separated string of team names.
    Returns "none" if the dictionary is empty.
    """
    team_names = list(teams.keys())
    return ", ".join(team_names) if team_names else "none"


def inquire_team(
    remote_provider: BasetenRemote, teams: Optional[Dict[str, Dict[str, str]]] = None
) -> Optional[str]:
    """
    Inquire for team selection if multiple teams are available.
    Returns team ID if selected, None otherwise.
    If teams is provided, uses that dict; otherwise fetches teams from API.
    """
    if teams is None:
        teams = remote_provider.api.get_teams()
    if len(teams) > 1:
        team_names = list(teams.keys())
        selected_team_name = inquirer.select(
            "👥 Which team do you want to use?", qmark="", choices=team_names
        ).execute()
        return get_team_id_from_name(teams, selected_team_name)
    # If 0 or 1 teams, return None (don't propagate team param)
    return None
