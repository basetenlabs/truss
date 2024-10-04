from pathlib import Path
from typing import Optional

DYNAMIC_CONFIG_MOUNT_DIR = "/etc/dynamic_config"


def get_dynamic_config_value(key: str) -> Optional[str]:
    dynamic_config_path = Path(DYNAMIC_CONFIG_MOUNT_DIR) / key
    if dynamic_config_path.exists() and dynamic_config_path.is_file():
        with dynamic_config_path.open() as dynamic_config_file:
            # Read and parse the file content assuming it's JSON
            dynamic_config_value = dynamic_config_file.read()
            return dynamic_config_value
    return None
