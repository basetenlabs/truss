from pathlib import Path
from typing import Optional

import aiofiles

DYNAMIC_CONFIG_MOUNT_DIR = "/etc/b10_dynamic_config"


def _read_file_sync(path: Path) -> str:
    with path.open() as f:
        return f.read()


async def _read_file_async(path: Path) -> str:
    async with aiofiles.open(path, "r") as f:
        return await f.read()


def get_dynamic_config_value_sync(key: str) -> Optional[str]:
    dynamic_config_path = Path(DYNAMIC_CONFIG_MOUNT_DIR) / key
    if dynamic_config_path.exists():
        return _read_file_sync(dynamic_config_path)
    return None


async def get_dynamic_config_file_path_async(key: str):
    dynamic_config_path = Path(DYNAMIC_CONFIG_MOUNT_DIR) / key
    return dynamic_config_path


async def get_dynamic_config_value_async(key: str) -> Optional[str]:
    dynamic_config_path = await get_dynamic_config_file_path_async(key)
    if dynamic_config_path.exists():
        return await _read_file_async(dynamic_config_path)
    return None
