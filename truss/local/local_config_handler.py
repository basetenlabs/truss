from pathlib import Path
from typing import Optional

from truss.local.local_config import LocalConfig


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
    def _config_path():
        return LocalConfigHandler.TRUSS_CONFIG_DIR / "config.yaml"

    @staticmethod
    def bptr_data_resolution_dir_path():
        bptr_data_dir = LocalConfigHandler.TRUSS_CONFIG_DIR / "bptr"
        bptr_data_dir.mkdir(exist_ok=True, parents=True)
        return bptr_data_dir

    @staticmethod
    def dynamic_config_path():
        dynamic_config_dir = LocalConfigHandler.TRUSS_CONFIG_DIR / "b10_dynamic_config"
        dynamic_config_dir.mkdir(exist_ok=True, parents=True)
        return dynamic_config_dir

    @staticmethod
    def set_dynamic_config(key: str, value: str):
        key_path = LocalConfigHandler.dynamic_config_path() / key
        with key_path.open("w") as key_file:
            key_file.write(value)

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
