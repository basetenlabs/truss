import copy
from dataclasses import replace
from pathlib import Path

from truss.local.local_config import LocalConfig
from truss.validation import validate_secret_name


class LocalConfigHandler:
    TRUSS_CONFIG_DIR = Path.home() / '.truss'

    @staticmethod
    def get_config() -> LocalConfig:
        if LocalConfigHandler._config_path().exists():
            return LocalConfig.from_yaml(LocalConfigHandler._config_path())
        return LocalConfig()

    @staticmethod
    def sync_secrets_mount_dir():
        """Syncs config secrets into a directory form, meant for mounting onto docker containers."""
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
            with secret_path.open('w') as secret_file:
                secret_file.write(secret_value)

    @staticmethod
    def set_secret(secret_name: str, secret_value: str):
        validate_secret_name(secret_name)
        LocalConfigHandler.TRUSS_CONFIG_DIR.mkdir(exist_ok=True, parents=True)
        local_config = LocalConfigHandler.get_config()
        new_secrets = {
            **local_config.secrets,
            secret_name: secret_value,
        }
        new_local_config = replace(local_config, secrets=new_secrets)
        new_local_config.write_to_yaml_file(LocalConfigHandler._config_path())

    @staticmethod
    def remove_secret(secret_name: str):
        LocalConfigHandler.TRUSS_CONFIG_DIR.mkdir(exist_ok=True, parents=True)
        local_config = LocalConfigHandler.get_config()
        new_secrets = copy.deepcopy(local_config.secrets)
        del new_secrets[secret_name]
        new_local_config = replace(local_config, secrets=new_secrets)
        new_local_config.write_to_yaml_file(LocalConfigHandler._config_path())

    @staticmethod
    def _config_dir():
        return LocalConfigHandler.TRUSS_CONFIG_DIR

    @staticmethod
    def _config_path():
        return LocalConfigHandler._config_dir() / 'config.yaml'

    @staticmethod
    def secrets_dir_path():
        return LocalConfigHandler._config_dir() / 'secrets'
