from typing import Optional

import click
from InquirerPy import inquirer
from InquirerPy.base.control import Choice
from InquirerPy.validator import ValidationError, Validator

from truss.base.constants import DEFAULT_REMOTE_NAME, DEFAULT_REMOTE_URL
from truss.cli.utils.common import check_is_interactive
from truss.cli.utils.output import console
from truss.remote.baseten import oauth
from truss.remote.baseten.api import resolve_rest_api_url
from truss.remote.baseten.custom_types import TeamType
from truss.remote.baseten.oauth import OAuthError
from truss.remote.remote_factory import USER_TRUSSRC_PATH, AuthType, RemoteFactory
from truss.remote.truss_remote import RemoteConfig


class NonEmptyValidator(Validator):
    def validate(self, document):
        if not document.text.strip():
            raise ValidationError(
                message="Please enter a non-empty value",
                cursor_position=len(document.text),
            )


def inquire_remote_config(
    *, remote_name: str = DEFAULT_REMOTE_NAME, remote_url: str = DEFAULT_REMOTE_URL
) -> RemoteConfig:
    # TODO(bola): extract questions from remote
    console.print("💻 Let's add a Baseten remote!")
    # If users need to adjust the remote url, they
    # can do so manually in the .trussrc file.
    method = inquirer.select(
        message="How would you like to authenticate?",
        qmark="",
        choices=[
            Choice(value="api_key", name="Paste an API key"),
            Choice(value="browser", name="Log in via browser (OAuth)"),
        ],
    ).execute()
    if method == "browser":
        try:
            credential = oauth.run_device_flow(resolve_rest_api_url(remote_url))
        except OAuthError as exc:
            raise click.ClickException(str(exc))
        return RemoteConfig(
            name=remote_name,
            configs={
                "remote_provider": DEFAULT_REMOTE_NAME,
                "remote_url": remote_url,
                "auth_type": AuthType.OAUTH,
                "oauth_access_token": credential.access_token,
                "oauth_refresh_token": credential.refresh_token,
                "oauth_expires_at": str(credential.expires_at),
            },
        )
    api_key = inquirer.secret(
        message="🤫 Quietly paste your API_KEY:", qmark="", validate=NonEmptyValidator()
    ).execute()
    return RemoteConfig(
        name=remote_name,
        configs={
            "remote_provider": DEFAULT_REMOTE_NAME,
            "auth_type": AuthType.API_KEY,
            "api_key": api_key,
            "remote_url": remote_url,
        },
    )


def inquire_remote_name(*, allow_create: bool = True) -> str:
    available_remotes = RemoteFactory.get_available_config_names()
    if len(available_remotes) == 0:
        if not allow_create:
            raise click.ClickException(
                "No remotes configured. Run `truss auth login` first."
            )
        if not check_is_interactive():
            raise click.UsageError(
                "No remote configured. Please configure a remote first "
                "(e.g. by running `truss login`)."
            )
        remote_config = inquire_remote_config()
        RemoteFactory.update_remote_config(remote_config)
        console.print(
            f"💾 Remote config `{remote_config.name}` saved to `{USER_TRUSSRC_PATH}`."
        )
        return remote_config.name
    elif len(available_remotes) == 1:
        return available_remotes[0]
    else:
        if not check_is_interactive():
            raise click.UsageError(
                "Multiple remotes available. Please specify one with --remote."
            )
        return inquirer.select(
            "🎮 Which remote do you want to connect to?",
            qmark="",
            choices=available_remotes,
        ).execute()


def inquire_model_name() -> str:
    return inquirer.text("📦 Name this model:", qmark="").execute()


def get_team_id_from_name(teams: dict[str, TeamType], team_name: str) -> Optional[str]:
    team = teams.get(team_name)
    return team.id if team else None


def format_available_teams(teams: dict[str, TeamType]) -> str:
    team_names = list(teams.keys())
    return ", ".join(team_names) if team_names else "none"


def inquire_team(
    existing_teams: Optional[dict[str, TeamType]] = None,
    prompt: str = "👥 Which team do you want to push to?",
) -> Optional[str]:
    if existing_teams is not None:
        # Sort with default team first, then alphanumerically (case-insensitive)
        sorted_teams = sorted(
            existing_teams.items(),
            key=lambda item: (not item[1].default, item[0].lower()),
        )

        # Create Choice objects: value is the actual team name, name is for display
        # This avoids issues with team names that contain "(default)"
        choices = [
            Choice(value=name, name=f"{name} (default)" if team.default else name)
            for name, team in sorted_teams
        ]

        # execute() returns the Choice.value (actual team name), not Choice.name
        return inquirer.select(prompt, qmark="", choices=choices).execute()

    # If no existing teams, return None (don't propagate team param)
    return None
