from contextlib import contextmanager
from pathlib import Path

from truss.local.local_config_handler import LocalConfigHandler

CONFIG_CONTENT = """
secrets:
    secret_name: secret_value
"""


def test_get_config(tmp_path):
    with _truss_config_dir(tmp_path, CONFIG_CONTENT):
        config = LocalConfigHandler.get_config()
        assert config.secrets["secret_name"] == "secret_value"


def test_set_secret(tmp_path):
    with _truss_config_dir(tmp_path, CONFIG_CONTENT):
        LocalConfigHandler.set_secret("another_secret_name", "another_secret_value")
        config = LocalConfigHandler.get_config()
        assert config.secrets["secret_name"] == "secret_value"
        assert config.secrets["another_secret_name"] == "another_secret_value"


def test_set_secret_override(tmp_path):
    with _truss_config_dir(tmp_path, CONFIG_CONTENT):
        LocalConfigHandler.set_secret("secret_name", "another_secret_value")
        config = LocalConfigHandler.get_config()
        assert config.secrets["secret_name"] == "another_secret_value"


def test_remove_secret(tmp_path):
    with _truss_config_dir(tmp_path, CONFIG_CONTENT):
        LocalConfigHandler.remove_secret("secret_name")
        config = LocalConfigHandler.get_config()
        assert "secret_name" not in config.secrets


def test_sync_secrets_mount_dir(tmp_path):
    with _truss_config_dir(tmp_path, CONFIG_CONTENT):
        LocalConfigHandler.sync_secrets_mount_dir()
        assert (tmp_path / "secrets").exists()
        with (tmp_path / "secrets" / "secret_name").open() as f:
            assert f.read() == "secret_value"


def test_signatures(tmp_path):
    with _truss_config_dir(tmp_path, CONFIG_CONTENT):
        assert LocalConfigHandler.get_signature("hash1") is None
        LocalConfigHandler.add_signature("hash1", "sig1")
        assert LocalConfigHandler.get_signature("hash1") == "sig1"


@contextmanager
def _truss_config_dir(path: Path, config_content: str):
    orig_config_dir = LocalConfigHandler.TRUSS_CONFIG_DIR
    LocalConfigHandler.TRUSS_CONFIG_DIR = path
    try:
        with (path / "config.yaml").open("w") as f:
            f.write(config_content)
        yield
    finally:
        LocalConfigHandler.TRUSS_CONFIG_DIR = orig_config_dir
