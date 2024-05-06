import os
from collections.abc import Mapping
from pathlib import Path
from typing import Dict, Optional

SECRETS_DOC_LINK = "https://truss.baseten.co/guides/secrets"


class SecretNotFound(Exception):
    pass


class SecretsResolver:
    SECRETS_MOUNT_DIR = "/secrets"
    SECRET_ENV_VAR_PREFIX = "TRUSS_SECRET_"

    @staticmethod
    def get_secrets(config: Dict):
        return Secrets(config.get("secrets", {}))

    @staticmethod
    def _resolve_secret(secret_name: str, default_value: Optional[str]):
        secret_value = default_value
        secret_env_var_name = SecretsResolver.SECRET_ENV_VAR_PREFIX + secret_name
        if secret_env_var_name in os.environ:
            secret_value = os.environ[secret_env_var_name]
        secret_path = SecretsResolver._secrets_mount_dir_path() / secret_name
        if secret_path.exists() and secret_path.is_file():
            with secret_path.open() as secret_file:
                secret_value = secret_file.read()
        return secret_value

    @staticmethod
    def _secrets_mount_dir_path():
        return Path(SecretsResolver.SECRETS_MOUNT_DIR)


class Secrets(Mapping):
    def __init__(self, base_secrets: Dict[str, str]):
        self._base_secrets = base_secrets

    def __getitem__(self, key: str) -> str:
        if key not in self._base_secrets:
            raise SecretNotFound(_secret_missing_error_message(key))

        found_secret = SecretsResolver._resolve_secret(key, self._base_secrets[key])
        if not found_secret:
            raise SecretNotFound(_secret_missing_error_message(key))

        return found_secret

    def __iter__(self):
        raise NotImplementedError(
            "Secrets are meant for lookup and can't be iterated on"
        )

    def __len__(self):
        return len(self._base_secrets)


def _secret_missing_error_message(key: str) -> str:
    return f"""
Secret '{key}' not found. Please ensure that:
  * Secret '{key}' is defined in the 'secrets' section of the Truss config file
  * The model was pushed with the --trusted flag
  * Secret '{key}' is defined in the secret manager
 Read more about secrets here: {SECRETS_DOC_LINK}.
    """
