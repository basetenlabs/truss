import copy
from dataclasses import replace
from pathlib import Path
from typing import Optional

from truss.local.local_config import LocalConfig
from truss.validation import validate_secret_name


class LocalConfigHandler:
    TRUSS_CONFIG_DIR = Path.home() / ".truss"

    @staticmethod
    def _ensure_config_dir():
        LocalConfigHandler.TRUSS_CONFIG_DIR.mkdir(exist_ok=True, parents=True)

    @staticmethod
    def get_config() -> LocalConfig:
        if LocalConfigHandler._config_path().exists():
            return LocalConfig.from_yaml(LocalConfigHandler._config_path())
        return LocalConfig()

    @staticmethod
    def sync_secrets_mount_dir():
        """Syncs config secrets into a directory form, meant for mounting onto docker containers."""
        LocalConfigHandler._ensure_config_dir()
        local_config = LocalConfigHandler.get_config()
        secrets = local_config.secrets
        secrets_dir = LocalConfigHandler.secrets_dir_path()
        if not secrets_dir.exists():
            secrets_dir.mkdir(parents=True)
        # Remove any files that don't correspond to a secret
        for path in secrets_dir.iterdir():
            if path.is_file() and path.name not in secrets:
                path.unlink()

        for secret_name, secret_value in secrets.items():
            secret_path = secrets_dir / secret_name
            with secret_path.open("w") as secret_file:
                secret_file.write(secret_value)

    @staticmethod
    def set_secret(secret_name: str, secret_value: str):
        LocalConfigHandler._ensure_config_dir()
        validate_secret_name(secret_name)
        local_config = LocalConfigHandler.get_config()
        new_secrets = {
            **local_config.secrets,
            secret_name: secret_value,
        }
        new_local_config = replace(local_config, secrets=new_secrets)
        new_local_config.write_to_yaml_file(LocalConfigHandler._config_path())

    @staticmethod
    def remove_secret(secret_name: str):
        LocalConfigHandler._ensure_config_dir()
        LocalConfigHandler.TRUSS_CONFIG_DIR.mkdir(exist_ok=True, parents=True)
        local_config = LocalConfigHandler.get_config()
        new_secrets = copy.deepcopy(local_config.secrets)
        del new_secrets[secret_name]
        new_local_config = replace(local_config, secrets=new_secrets)
        new_local_config.write_to_yaml_file(LocalConfigHandler._config_path())

    @staticmethod
    def _config_path():
        return LocalConfigHandler.TRUSS_CONFIG_DIR / "config.yaml"

    @staticmethod
    def secrets_dir_path():
        return LocalConfigHandler.TRUSS_CONFIG_DIR / "secrets"

    @staticmethod
    def _signatures_dir_path():
        return LocalConfigHandler.TRUSS_CONFIG_DIR / "signatures"

    @staticmethod
    def shadow_trusses_dir_path():
        return LocalConfigHandler.TRUSS_CONFIG_DIR / "shadow_trusses"

    @staticmethod
    def add_signature(truss_hash: str, signature: str):
        if truss_hash is None:
            raise ValueError("truss_hash is None")

        LocalConfigHandler._ensure_config_dir()
        signature_dir = LocalConfigHandler._signatures_dir_path()
        signature_dir.mkdir(exist_ok=True)
        with (signature_dir / truss_hash).open("w") as signature_file:
            signature_file.write(signature)

    @staticmethod
    def get_signature(truss_hash: str) -> Optional[str]:
        if truss_hash is None:
            raise ValueError("truss_hash is None")

        LocalConfigHandler._ensure_config_dir()
        signature_dir = LocalConfigHandler._signatures_dir_path()
        signature_dir.mkdir(exist_ok=True)
        signature_file_path = signature_dir / truss_hash
        if not signature_file_path.exists() or not signature_file_path.is_file():
            return None

        with signature_file_path.open() as signature_file:
            return signature_file.read()
