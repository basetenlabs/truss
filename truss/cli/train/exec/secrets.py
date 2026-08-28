"""`--env` / `--secret` handling for `truss train exec`."""

import logging
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union

import rich_click as click

from truss.remote.baseten.api import BasetenApi
from truss_train.definitions import SecretReference

logger = logging.getLogger(__name__)

# There is no CLI command to create a workspace secret, so the settings page is the
# only actionable next step we can point at.
SECRETS_SETTINGS_URL = "https://app.baseten.co/settings/secrets"


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
    for key, value in entries:
        if key in environment_variables:
            raise click.UsageError(
                f"Environment variable '{key}' is set more than once by "
                "--env / --secret."
            )
        environment_variables[key] = value
    return environment_variables


def _known_secret_names(response: Any) -> Optional[Set[str]]:
    """Secret names from a `GET v1/secrets` payload, or None if it is unrecognized.

    `get_all_secrets` had no callers before this, and the response shape is not
    pinned down by any test or doc in this repo, so accept the plausible shapes and
    give up rather than guess -- returning None means "don't check", which is very
    different from returning an empty set.
    """
    if isinstance(response, dict):
        entries = response.get("secrets")
    elif isinstance(response, list):
        entries = response
    else:
        return None
    if not isinstance(entries, list):
        return None

    names: Set[str] = set()
    for entry in entries:
        if isinstance(entry, str):
            names.add(entry)
        elif isinstance(entry, dict) and isinstance(entry.get("name"), str):
            names.add(entry["name"])
        else:
            # An unfamiliar entry shape would mean guessing, and a wrong guess now
            # fails the command rather than just warning.
            return None
    return names


def validate_secret_references(
    api: BasetenApi, environment_variables: Mapping[str, Union[str, SecretReference]]
) -> None:
    """Fail before pushing if a `--secret` names a secret the workspace doesn't have.

    Two cases, deliberately treated differently:

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

    try:
        response = api.get_all_secrets()
    except Exception:
        logger.debug("Could not list workspace secrets; skipping check.", exc_info=True)
        return

    # Outside the try: a bug in the parser should surface, not be mistaken for an
    # unreachable API.
    known = _known_secret_names(response)
    if known is None:
        logger.debug("Unrecognized v1/secrets payload; skipping check.")
        return

    missing = [name for name in referenced if name not in known]
    if not missing:
        return

    plural = len(missing) > 1
    raise click.UsageError(
        f"{'Secrets' if plural else 'Secret'} {', '.join(missing)} "
        f"{'were' if plural else 'was'} not found in this workspace's secrets. "
        f"Create {'them' if plural else 'it'} at {SECRETS_SETTINGS_URL}. "
        "(The listing this checks against is not team-scoped, so if the secret does "
        "exist for the team this job runs in, please report the mismatch.)"
    )
