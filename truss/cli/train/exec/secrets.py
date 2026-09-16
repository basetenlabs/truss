"""`--env` / `--secret` handling for `truss train exec`."""

import logging
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union

import rich_click as click

from truss.remote.baseten.api import BasetenApi
from truss.remote.baseten.custom_types import APIKeyCategory
from truss_train.definitions import SecretReference

logger = logging.getLogger(__name__)

# There is no CLI command to create a workspace secret, so the settings page is the
# only actionable next step we can point at.
SECRETS_SETTINGS_URL = "https://app.baseten.co/settings/secrets"

# The variable the Baseten SDKs read their credential from.
BASETEN_API_KEY_ENV_VAR = "BASETEN_API_KEY"

# The narrowest type that works. Both the training control plane and the trainer
# inference host require MANAGE_TRAINING, which no invoke-scoped key type is granted,
# and a key carries exactly one type. The team is what actually confines the key: its
# permissions are object-level on that one team, so it cannot reach another team's
# resources.
_PROVISIONED_KEY_CATEGORY = APIKeyCategory.WORKSPACE_MANAGE_ALL


def _parse_key_value_flag(
    flag: str, expected: str, entry: str, require_value: bool = False
) -> Tuple[str, str]:
    # partition, not split: a value may itself contain `=`.
    key, separator, value = entry.partition("=")
    # An empty --env value is legitimate; an empty secret *name* is not.
    if not separator or not key or (require_value and not value):
        raise click.UsageError(f"Invalid {flag} value '{entry}'. Expected {expected}.")
    return key, value


def parse_environment_variables(
    env: Sequence[str] = (), secrets: Sequence[str] = ()
) -> Dict[str, Union[str, SecretReference]]:
    """Turn `--env KEY=VALUE` and `--secret KEY=SECRET_NAME` flags into the
    `Runtime.environment_variables` mapping."""
    entries: List[Tuple[str, Union[str, SecretReference]]] = []
    for entry in env:
        key, value = _parse_key_value_flag("--env", "KEY=VALUE", entry)
        entries.append((key, value))
    for entry in secrets:
        key, secret_name = _parse_key_value_flag(
            "--secret", "KEY=SECRET_NAME", entry, require_value=True
        )
        entries.append((key, SecretReference(name=secret_name)))

    environment_variables: Dict[str, Union[str, SecretReference]] = {}
    for key, resolved in entries:
        if key in environment_variables:
            raise click.UsageError(
                f"Environment variable '{key}' is set more than once by "
                "--env / --secret."
            )
        environment_variables[key] = resolved
    return environment_variables


def _known_secret_names(response: Any) -> Optional[Set[str]]:
    """Secret names from a secrets listing, or None if the payload is unrecognized.

    `{"secrets": [{"name": ...}, ...]}` is the documented shape of both listing
    endpoints. None means "don't check", which is very different from an empty set,
    so anything that doesn't match gives up rather than reporting every name missing.
    """
    if not isinstance(response, dict):
        return None
    entries = response.get("secrets")
    if not isinstance(entries, list):
        return None
    if not all(
        isinstance(entry, dict) and isinstance(entry.get("name"), str)
        for entry in entries
    ):
        return None
    return {entry["name"] for entry in entries}


def validate_secret_references(
    api: BasetenApi,
    environment_variables: Mapping[str, Union[str, SecretReference]],
    team_id: Optional[str] = None,
) -> None:
    """Fail before pushing if a `--secret` names a secret the job's team doesn't have.

    The server resolves a `SecretReference` against the training project's team, so
    that is the scope to check. Without a `team_id` the job lands in the
    organization's default team, which the caller-wide listing cannot isolate, so the
    check falls back to "does this name exist in any team I can see" -- a superset,
    which can only miss a failure, never invent one.

    Two outcomes, deliberately treated differently:

    * The listing came back and the name isn't in it -> hard error. The job would
      fail to start, so failing here is faster and clearer.
    * The listing call failed, or returned something we can't parse -> continue. That
      is an API problem, not evidence the secret is missing, and a convenience check
      must not break the command over an API blip or a permissions quirk.
    """
    referenced = sorted(
        {
            value.name
            for value in environment_variables.values()
            if isinstance(value, SecretReference)
        }
    )
    if not referenced:
        # No --secret flags, so don't spend a round trip on the common path.
        return

    scope = "team" if team_id else "workspace"
    try:
        response = api.get_team_secrets(team_id) if team_id else api.get_all_secrets()
    except Exception:
        logger.debug("Could not list %s secrets; skipping check.", scope, exc_info=True)
        return

    # Outside the try: a bug in the parser should surface, not be mistaken for an
    # unreachable API.
    known = _known_secret_names(response)
    if known is None:
        logger.debug("Unrecognized %s secrets payload; skipping check.", scope)
        return

    missing = [name for name in referenced if name not in known]
    if not missing:
        return

    plural = len(missing) > 1
    raise click.UsageError(
        f"{'Secrets' if plural else 'Secret'} {', '.join(missing)} "
        f"{'were' if plural else 'was'} not found in this {scope}'s secrets. "
        f"Create {'them' if plural else 'it'} at {SECRETS_SETTINGS_URL}."
    )


def team_api_key_secret_name(team_id: str) -> str:
    """The secret holding the Baseten credential jobs in `team_id` run with.

    One name per team, so a second run reuses the first run's key instead of minting
    another, and two teams never share a credential.
    """
    return f"truss-team-{team_id}-exec-api-key"


def ensure_team_api_key_secret(
    api: BasetenApi, team_id: str
) -> Optional[SecretReference]:
    """A reference to the team's Baseten credential secret, minted on first use.

    Returns None if the secret could neither be found nor created -- the command
    still pushes in that case, because a job that makes no Baseten API calls does not
    need a credential at all.

    A key's plaintext is returned only when it is created, so an existing secret is
    reused as-is rather than refreshed: there is nothing to compare it against.
    """
    name = team_api_key_secret_name(team_id)
    try:
        known = _known_secret_names(api.get_team_secrets(team_id))
    except Exception:
        logger.debug("Could not list team secrets.", exc_info=True)
        return None
    if known is None:
        # Minting against an unreadable listing risks replacing a working key.
        logger.debug("Unrecognized team secrets payload.")
        return None
    if name in known:
        return SecretReference(name=name)

    try:
        response = api.create_api_key(_PROVISIONED_KEY_CATEGORY, name, team_id=team_id)
        value = response.get("api_key") if isinstance(response, dict) else None
        if not value:
            logger.debug("No api_key in the create response.")
            return None
        api.upsert_team_secret(team_id, name, value)
    except Exception:
        logger.debug("Could not provision the team api key.", exc_info=True)
        return None
    return SecretReference(name=name)
