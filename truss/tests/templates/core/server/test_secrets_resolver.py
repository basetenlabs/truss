import os
from contextlib import contextmanager
from pathlib import Path

import pytest

from truss.templates.shared.secrets_resolver import SecretNotFound, SecretsResolver

CONFIG = {"secrets": {"secret_key": "default_secret_value"}}


def test_resolve_default_value():
    secrets = SecretsResolver.get_secrets(CONFIG)
    assert secrets["secret_key"] == "default_secret_value"


def test_resolve_env_var():
    secrets = SecretsResolver.get_secrets(CONFIG)
    with _override_env_var("TRUSS_SECRET_secret_key", "secret_value_from_env"):
        assert secrets["secret_key"] == "secret_value_from_env"


def test_resolve_mounted_secrets(tmp_path):
    secrets = SecretsResolver.get_secrets(CONFIG)
    with (tmp_path / "secret_key").open("w") as f:
        f.write("secret_value_from_mounted_secrets")
    with (
        _secrets_mount_dir(tmp_path),
        _override_env_var("TRUSS_SECRET_secret_key", "secret_value_from_env"),
    ):
        assert secrets["secret_key"] == "secret_value_from_mounted_secrets"


def test_resolve_list_mounted_secrets(tmp_path):
    secrets = SecretsResolver.get_secrets({"secrets": ["secret_key", "missing_key"]})
    with (tmp_path / "secret_key").open("w") as f:
        f.write("secret_value_from_mounted_secrets")
    with _secrets_mount_dir(tmp_path):
        assert secrets["secret_key"] == "secret_value_from_mounted_secrets"
        with pytest.raises(SecretNotFound):
            secrets["missing_key"]
        with pytest.raises(SecretNotFound):
            secrets["undeclared_key"]


def test_contains_and_get_for_missing_secrets(tmp_path):
    secrets = SecretsResolver.get_secrets(
        {"secrets": {"secret_key": "default_secret_value", "no_value_key": None}}
    )
    with _secrets_mount_dir(tmp_path):
        assert "secret_key" in secrets
        assert secrets.get("secret_key") == "default_secret_value"
        for key in ("no_value_key", "undeclared_key"):
            assert key not in secrets
            assert secrets.get(key) is None
            assert secrets.get(key, "fallback") == "fallback"


def test_contains_and_get_resolve_mounted_secrets(tmp_path):
    secrets = SecretsResolver.get_secrets({"secrets": ["secret_key"]})
    with (tmp_path / "secret_key").open("w") as f:
        f.write("secret_value_from_mounted_secrets")
    with _secrets_mount_dir(tmp_path):
        assert "secret_key" in secrets
        assert secrets.get("secret_key") == "secret_value_from_mounted_secrets"


@contextmanager
def _secrets_mount_dir(path: Path):
    orig_secrets_mount_dir = SecretsResolver.SECRETS_MOUNT_DIR
    SecretsResolver.SECRETS_MOUNT_DIR = str(path)
    try:
        yield
    finally:
        SecretsResolver.SECRETS_MOUNT_DIR = orig_secrets_mount_dir


@contextmanager
def _override_env_var(key: str, value: str):
    orig_value = os.environ.get(key, None)
    try:
        os.environ[key] = value
        yield
    finally:
        if orig_value is not None:
            os.environ[key] = orig_value
        else:
            del os.environ[key]
