from pathlib import Path
from typing import Optional


class ExtraConfigResolver:
    EXTRA_CONFIG_MOUNT_DIR = "/etc/extra_config"

    @staticmethod
    def get_config_value(config_key_name: str) -> Optional[str]:
        extra_config_path = (
            ExtraConfigResolver._extra_config_mount_dir_path() / config_key_name
        )
        if extra_config_path.exists() and extra_config_path.is_file():
            with extra_config_path.open() as extra_config_file:
                extra_config_value = extra_config_file.read()
                return extra_config_value
        return None

    @staticmethod
    def _extra_config_mount_dir_path():
        return Path(ExtraConfigResolver.EXTRA_CONFIG_MOUNT_DIR)


# Implement caching of config KV pairs
