import enum
import inspect
import json
import logging
import os

try:
    from configparser import DEFAULTSECT, ConfigParser  # type: ignore
except ImportError:
    # We need to do this for old python.
    from configparser import DEFAULTSECT
    from configparser import SafeConfigParser as ConfigParser


from functools import partial
from operator import is_not
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type

import keyring
import keyring.backends.fail
import keyring.backends.null
import keyring.errors

from truss.remote.baseten import BasetenRemote
from truss.remote.truss_remote import RemoteConfig, TrussRemote

logger = logging.getLogger(__name__)

KEYRING_SERVICE = "baseten-truss"
KEYRING_DISABLED_ENV = "BASETEN_TRUSS_AUTH_KEYRING_DISABLED"


class AuthType(str, enum.Enum):
    API_KEY = "api_key"
    OAUTH = "oauth"

    def __str__(self) -> str:
        return self.value


_INLINE_SECRET_KEYS_BY_AUTH_TYPE: Dict[str, Tuple[str, ...]] = {
    AuthType.API_KEY: ("api_key",),
    AuthType.OAUTH: ("oauth_access_token", "oauth_refresh_token", "oauth_expires_at"),
}
_INLINE_SECRET_KEYS = tuple(
    key for keys in _INLINE_SECRET_KEYS_BY_AUTH_TYPE.values() for key in keys
)
_keyring_fallback_warned = False

USER_TRUSSRC_PATH = Path(os.environ.get("USER_TRUSSRC_PATH", "~/.trussrc")).expanduser()


def load_config() -> ConfigParser:
    config = ConfigParser()
    config.read(USER_TRUSSRC_PATH)
    return config


def update_config(config: ConfigParser):
    with open(USER_TRUSSRC_PATH, "w") as configfile:
        config.write(configfile)


class RemoteFactory:
    """
    A factory for instantiating a TrussRemote from a .trussrc file and a user-specified remote config name
    """

    REGISTRY: Dict[str, Type[TrussRemote]] = {"baseten": BasetenRemote}

    @staticmethod
    def get_available_config_names() -> List[str]:
        if not USER_TRUSSRC_PATH.exists():
            return []

        config = load_config()
        return list(filter(partial(is_not, DEFAULTSECT), config.keys()))

    @staticmethod
    def load_remote_config(remote_name: str) -> RemoteConfig:
        """
        Load and validate a remote config from the .trussrc file
        """
        if not USER_TRUSSRC_PATH.exists():
            raise FileNotFoundError("No ~/.trussrc file found.")

        config = load_config()

        if remote_name not in config:
            raise ValueError(f"Service provider {remote_name} not found in ~/.trussrc")

        remote_config = RemoteConfig(
            name=remote_name, configs=dict(config[remote_name])
        )
        _apply_secrets_to_config(remote_config)
        return remote_config

    @staticmethod
    def update_remote_config(remote_config: RemoteConfig):
        """
        Load and validate a remote config from the .trussrc file
        """
        remote_config = RemoteConfig(
            name=remote_config.name, configs=dict(remote_config.configs)
        )
        _offload_secrets_from_config(remote_config)
        config = load_config()
        config[remote_config.name] = remote_config.configs
        update_config(config)

    @staticmethod
    def get_remote_team(remote_name: str) -> Optional[str]:
        """
        Get team name from remote config if configured.
        """
        try:
            config = RemoteFactory.load_remote_config(remote_name)
            return config.configs.get("team")
        except (FileNotFoundError, ValueError):
            return None

    @staticmethod
    def remove_remote_config(remote_name: str) -> None:
        """Remove a remote entirely: keyring entry and trussrc section.

        ``truss auth login`` adds a remote section; ``truss auth logout`` is the
        inverse. Caller is expected to have verified the remote exists.
        """
        if not _keyring_disabled_by_env() and _keyring_backend_usable():
            try:
                keyring.delete_password(KEYRING_SERVICE, remote_name)
            except keyring.errors.PasswordDeleteError:
                pass
            except keyring.errors.KeyringError as exc:
                logger.warning("Keyring delete failed for %s: %s", remote_name, exc)
        if not USER_TRUSSRC_PATH.exists():
            return
        config = load_config()
        if remote_name in config:
            config.remove_section(remote_name)
            update_config(config)

    @classmethod
    def create(cls, remote: str) -> TrussRemote:
        remote_config = cls.load_remote_config(remote).configs
        remote_config.setdefault("oauth_remote_name", remote)
        if "remote_provider" not in remote_config:
            raise ValueError(f"Missing 'remote_provider' field for remote `{remote}`.")
        provider = remote_config.pop("remote_provider")
        if provider not in cls.REGISTRY:
            raise ValueError(f"Remote provider {provider} not found in registry.")
        remote_class = cls.REGISTRY[provider]

        parameters = inspect.signature(remote_class.__init__).parameters
        init_params = {n for n in parameters if n not in {"self", "args", "kwargs"}}
        required_params = {
            n
            for n, p in parameters.items()
            if p.default == inspect.Parameter.empty
            and n not in {"self", "args", "kwargs"}
        }
        accepts_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()
        )

        # If class accepts **kwargs, pass everything, else only known params
        if accepts_kwargs:
            passed_config = remote_config
        else:
            passed_config = {k: v for k, v in remote_config.items() if k in init_params}

        missing = required_params - set(remote_config.keys())
        if missing:
            raise ValueError(
                f"Missing required parameter(s) {list(missing)} for remote `{remote}`."
            )
        return remote_class(**passed_config)


def _keyring_disabled_by_env() -> bool:
    return os.environ.get(KEYRING_DISABLED_ENV, "").lower() in ("1", "true", "yes")


def _keyring_backend_usable() -> bool:
    backend = keyring.get_keyring()
    return not isinstance(
        backend, (keyring.backends.fail.Keyring, keyring.backends.null.Keyring)
    )


def _apply_secrets_to_config(remote_config: RemoteConfig) -> None:
    """Merge any keyring-stored secrets into ``remote_config.configs``.

    Sections without ``auth_type`` (legacy plaintext) are returned unchanged.
    Sections opted into ``auth_type`` get their secret keys filled from the
    keyring entry under (service=baseten-truss, account=remote_name). Inline
    secrets already present on the section win (env-disabled or
    backend-unavailable fallback). If neither inline nor keyring can supply
    all expected secrets, raises so callers see a clear failure.
    """
    configs = remote_config.configs
    auth_type = configs.get("auth_type")
    if not isinstance(auth_type, str):
        return
    secret_keys = _INLINE_SECRET_KEYS_BY_AUTH_TYPE.get(auth_type)
    if secret_keys is None:
        return
    if all(key in configs for key in secret_keys):
        return
    if _keyring_disabled_by_env() or not _keyring_backend_usable():
        raise ValueError(
            f"No credentials for remote {remote_config.name!r}: keyring is "
            "unavailable and no inline secret is present."
        )
    try:
        blob = keyring.get_password(KEYRING_SERVICE, remote_config.name)
    except keyring.errors.KeyringError as exc:
        raise ValueError(
            f"Keyring read failed for remote {remote_config.name!r}: {exc}"
        ) from exc
    if not blob:
        raise ValueError(
            f"No credentials in keyring for remote {remote_config.name!r}; "
            "run `truss login`."
        )
    try:
        payload = json.loads(blob)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Keyring entry for remote {remote_config.name!r} is not valid JSON."
        ) from exc
    if payload.get("auth_type") != auth_type or not all(
        key in payload for key in secret_keys
    ):
        raise ValueError(
            f"Keyring entry for remote {remote_config.name!r} is malformed."
        )
    for key in secret_keys:
        configs.setdefault(key, payload[key])


def _offload_secrets_from_config(remote_config: RemoteConfig) -> None:
    """If the section opts into ``auth_type``, push secrets into the keyring.

    Only triggers when the caller has set ``auth_type`` on ``configs``. Legacy
    callers writing plaintext ``api_key`` (no ``auth_type``) flow through
    unchanged. When the keyring is disabled by env var, the secrets are left
    inline silently (the user opted in to that). When the keyring backend is
    unavailable or a write fails, we warn once and leave the secrets inline.
    """
    global _keyring_fallback_warned
    configs = remote_config.configs
    auth_type = configs.get("auth_type")
    if not isinstance(auth_type, str):
        return
    secret_keys = _INLINE_SECRET_KEYS_BY_AUTH_TYPE.get(auth_type)
    if secret_keys is None or not all(key in configs for key in secret_keys):
        return
    if _keyring_disabled_by_env():
        return
    if not _keyring_backend_usable():
        if not _keyring_fallback_warned:
            logger.warning(
                "Warning: no usable OS keyring backend; storing credentials for "
                "%s in plaintext in %s.",
                remote_config.name,
                USER_TRUSSRC_PATH,
            )
            _keyring_fallback_warned = True
        return
    payload = json.dumps(
        {"auth_type": auth_type, **{key: configs[key] for key in secret_keys}}
    )
    try:
        keyring.set_password(KEYRING_SERVICE, remote_config.name, payload)
    except keyring.errors.KeyringError as exc:
        logger.warning(
            "Warning: keyring write failed for %s (%s); leaving secret in plaintext.",
            remote_config.name,
            exc,
        )
        return
    for key in _INLINE_SECRET_KEYS:
        configs.pop(key, None)
