from contextlib import contextmanager
from pathlib import Path

from truss.local.local_config_handler import LocalConfigHandler

CONFIG_CONTENT = """
use_sudo: true
"""


def test_get_config(tmp_path):
    with _truss_config_dir(tmp_path, CONFIG_CONTENT):
        assert LocalConfigHandler.get_config().use_sudo


def test_get_config_ignores_legacy_secrets(tmp_path):
    config_content = """
use_sudo: true
secrets:
    secret_name: secret_value
"""
    with _truss_config_dir(tmp_path, config_content):
        assert LocalConfigHandler.get_config().use_sudo


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
