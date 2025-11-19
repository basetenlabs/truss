from typing import Optional

from InquirerPy import inquirer
from InquirerPy.validator import ValidationError, Validator

from truss.cli.utils.output import console
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


def get_team_id_from_name(
    teams: dict[str, dict[str, str]], team_name: str
) -> Optional[str]:
    team = teams.get(team_name)
    return team["id"] if team else None


def format_available_teams(teams: dict[str, dict[str, str]]) -> str:
    team_names = list(teams.keys())
    return ", ".join(team_names) if team_names else "none"


def inquire_team(
    existing_teams: Optional[dict[str, dict[str, str]]] = None,
) -> Optional[str]:
    if existing_teams is not None:
        selected_team_name = inquirer.select(
            "👥 Which team do you want to push to?",
            qmark="",
            choices=list[str](existing_teams.keys()),
        ).execute()
        return selected_team_name

    # If no existing teams, return None (don't propagate team param)
    return None
