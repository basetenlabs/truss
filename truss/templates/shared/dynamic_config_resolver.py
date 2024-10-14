from pathlib import Path
from typing import Optional

import aiofiles

DYNAMIC_CONFIG_MOUNT_DIR = "/etc/b10_dynamic_config"


def get_dynamic_config_value_sync(key: str) -> Optional[str]:
    dynamic_config_path = Path(DYNAMIC_CONFIG_MOUNT_DIR) / key
    if dynamic_config_path.exists():
        with dynamic_config_path.open() as dynamic_config_file:
            return dynamic_config_file.read()
    return None


async def get_dynamic_config_file_path_async(key: str):
    dynamic_config_path = Path(DYNAMIC_CONFIG_MOUNT_DIR) / key
    return dynamic_config_path


async def get_dynamic_config_value_async(key: str) -> Optional[str]:
    dynamic_config_path = await get_dynamic_config_file_path_async(key)
    if dynamic_config_path.exists():
        async with aiofiles.open(dynamic_config_path, "r") as dynamic_config_file:
            return await dynamic_config_file.read()
    return None
