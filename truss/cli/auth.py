"""`truss auth` command group: login, logout, status."""

from typing import Optional

import rich_click as click

from truss.base.constants import DEFAULT_REMOTE_NAME, DEFAULT_REMOTE_URL
from truss.cli.remote_cli import inquire_remote_config, inquire_remote_name
from truss.cli.utils import common
from truss.cli.utils.common import check_is_interactive
from truss.cli.utils.output import console
from truss.remote.baseten import oauth
from truss.remote.baseten.api import resolve_rest_api_url
from truss.remote.baseten.oauth import OAuthCredential, OAuthError
from truss.remote.remote_factory import (
    _INLINE_SECRET_KEYS,
    USER_TRUSSRC_PATH,
    AuthType,
    RemoteFactory,
    load_config,
)
from truss.remote.truss_remote import RemoteConfig


@click.group(name="auth")
def auth_group() -> None:
    """Manage authentication."""


@auth_group.command(name="login")
@click.option("--browser", is_flag=True, help="Log in via browser (OAuth device flow).")
@click.option(
    "--api-key", "api_key", type=str, default=None, help="API key for authentication."
)
@click.option("--remote", type=str, default=None, help="Remote name to create.")
@common.common_options()
def auth_login(browser: bool, api_key: Optional[str], remote: Optional[str]) -> None:
    """Log in to a Baseten remote."""
    do_login(browser=browser, api_key=api_key, remote=remote)


@auth_group.command(name="logout")
@click.option(
    "--remote",
    type=str,
    default=None,
    help="Remote name. Inferred when only one is configured.",
)
@common.common_options()
def auth_logout(remote: Optional[str]) -> None:
    """Log out of a Baseten remote and remove it from the trussrc."""
    remote_name = remote or inquire_remote_name(allow_create=False)
    try:
        cfg = RemoteFactory.load_remote_config(remote_name)
    except (FileNotFoundError, ValueError) as exc:
        raise click.ClickException(f"Not logged in to remote {remote_name!r}: {exc}")

    if cfg.configs.get("auth_type") == AuthType.OAUTH:
        credential = OAuthCredential(
            access_token=cfg.configs["oauth_access_token"],
            refresh_token=cfg.configs["oauth_refresh_token"],
            expires_at=int(cfg.configs["oauth_expires_at"]),
        )
        oauth.revoke(resolve_rest_api_url(cfg.configs["remote_url"]), credential)

    RemoteFactory.remove_remote_config(remote_name)
    console.print(f"👋 Logged out of remote `{remote_name}`.")


@auth_group.command(name="status")
@click.option(
    "--remote",
    type=str,
    default=None,
    help="Remote name. Inferred when only one is configured.",
)
@common.common_options()
def auth_status(remote: Optional[str]) -> None:
    """Show the active authentication for a remote."""
    remote_name = remote or inquire_remote_name(allow_create=False)
    if not USER_TRUSSRC_PATH.exists():
        raise click.ClickException(f"No trussrc found at {USER_TRUSSRC_PATH}.")
    raw = load_config()
    if remote_name not in raw:
        raise click.ClickException(f"No remote {remote_name!r} configured.")
    section = raw[remote_name]
    auth_type = section.get("auth_type")
    if not auth_type:
        if "api_key" in section:
            auth_type = "api_key (legacy plaintext)"
            source = "trussrc-inline"
        else:
            raise click.ClickException(
                f"Remote {remote_name!r} has no credentials configured."
            )
    elif any(key in section for key in _INLINE_SECRET_KEYS):
        source = "trussrc-inline"
    else:
        source = "keyring"
    remote_url = section.get("remote_url", "(unset)")
    console.print(f"remote: {remote_name}")
    console.print(f"remote_url: {remote_url}")
    console.print(f"auth_type: {auth_type}")
    console.print(f"source: {source}")


def do_login(
    *, browser: bool, api_key: Optional[str], remote: Optional[str] = None
) -> None:
    """Shared implementation for `truss login` and `truss auth login`."""
    if browser and api_key is not None:
        raise click.UsageError("--browser and --api-key are mutually exclusive.")

    remote_name = remote or DEFAULT_REMOTE_NAME
    remote_url = _existing_remote_url(remote_name) or DEFAULT_REMOTE_URL

    if api_key is not None:
        RemoteFactory.update_remote_config(
            RemoteConfig(
                name=remote_name,
                configs={
                    "remote_provider": DEFAULT_REMOTE_NAME,
                    "remote_url": remote_url,
                    "auth_type": AuthType.API_KEY,
                    "api_key": api_key,
                },
            )
        )
    elif browser:
        try:
            credential = oauth.run_device_flow(resolve_rest_api_url(remote_url))
        except OAuthError as exc:
            raise click.ClickException(str(exc))
        RemoteFactory.update_remote_config(
            RemoteConfig(
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
        )
    else:
        if not check_is_interactive():
            raise click.UsageError(
                "Specify --browser or --api-key when running non-interactively."
            )
        config = inquire_remote_config(remote_name=remote_name, remote_url=remote_url)
        RemoteFactory.update_remote_config(config)
        remote_name = config.name
    console.print(f"🔓 Logged in to remote `{remote_name}`.")


def _existing_remote_url(remote_name: str) -> Optional[str]:
    if not USER_TRUSSRC_PATH.exists():
        return None
    raw = load_config()
    if remote_name not in raw:
        return None
    return raw[remote_name].get("remote_url")
