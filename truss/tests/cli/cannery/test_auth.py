import os
from pathlib import Path
from unittest.mock import Mock

import click
import pytest

from truss.cli.cannery.auth import (
    BasetenExchangeAuthProvider,
    CanneryCredential,
    ExchangedToken,
    ExplicitTokenFileAuthProvider,
    LoopbackNoAuthProvider,
    child_environment,
)
from truss.cli.cannery.config import ActiveRemote
from truss.remote.truss_remote import RemoteConfig


def _active_remote() -> ActiveRemote:
    return ActiveRemote(
        name="baseten",
        remote_url="https://app.baseten.co",
        config=RemoteConfig(
            name="baseten",
            configs={
                "remote_provider": "baseten",
                "remote_url": "https://app.baseten.co",
                "api_key": "secret-api-key",
            },
        ),
    )


def test_loopback_provider_has_no_token_file():
    with LoopbackNoAuthProvider().acquire("correlation") as credential:
        assert credential.token_file is None
        assert credential.mechanism == "loopback-no-auth"


def test_explicit_token_file_requires_owner_only_permissions(tmp_path):
    token_file = tmp_path / "token"
    token_file.write_text("secret")
    token_file.chmod(0o640)

    with pytest.raises(click.UsageError, match="owner-only"):
        ExplicitTokenFileAuthProvider(token_file).acquire("correlation")


def test_explicit_token_file_rejects_symlink(tmp_path):
    token_file = tmp_path / "token"
    token_file.write_text("secret")
    token_file.chmod(0o600)
    token_link = tmp_path / "token-link"
    token_link.symlink_to(token_file)

    with pytest.raises(click.UsageError, match="symlink"):
        ExplicitTokenFileAuthProvider(token_link).acquire("correlation")


def test_exchange_token_is_private_refreshable_and_cleaned_up():
    adapter = Mock()
    adapter.exchange.side_effect = [
        ExchangedToken("first-secret"),
        ExchangedToken("second-secret"),
    ]
    provider = BasetenExchangeAuthProvider(adapter, _active_remote())

    with provider.acquire("corr-123") as credential:
        assert credential.token_file is not None
        token_file = credential.token_file
        token_directory = token_file.parent
        assert token_file.read_text() == "first-secret"
        assert token_file.stat().st_mode & 0o777 == 0o600
        assert token_directory.stat().st_mode & 0o777 == 0o700

        original_inode = token_file.stat().st_ino
        credential.refresh()

        assert token_file.read_text() == "second-secret"
        assert token_file.stat().st_ino != original_inode
        assert not list(token_directory.glob(".token-*.tmp"))

    assert not token_directory.exists()
    assert adapter.exchange.call_args_list[0].args[1] == "corr-123"
    assert adapter.exchange.call_args_list[1].args[1] == "corr-123"


def test_exchange_failure_does_not_include_adapter_secret():
    adapter = Mock()
    adapter.exchange.side_effect = RuntimeError("failed with bearer top-secret")
    provider = BasetenExchangeAuthProvider(adapter, _active_remote())

    with pytest.raises(click.ClickException) as exc_info:
        provider.acquire("correlation")

    assert "top-secret" not in str(exc_info.value)


def test_child_environment_preserves_proxy_but_removes_parent_credentials(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example")
    monkeypatch.setenv("BASETEN_API_KEY", "parent-secret")
    monkeypatch.setenv("CANNERY_AUTH_TOKEN", "stale-secret")
    token_file = tmp_path / "token"
    credential = CanneryCredential(token_file, "test")

    env = child_environment(credential, "org", "corr-123", tmp_path / "diagnostic")

    assert env["HTTPS_PROXY"] == "http://proxy.example"
    assert "BASETEN_API_KEY" not in env
    assert "CANNERY_AUTH_TOKEN" not in env
    assert env["CANNERY_AUTH_TOKEN_FILE"] == str(token_file)


@pytest.mark.skipif(not hasattr(os, "getuid"), reason="Unix ownership check")
def test_explicit_token_file_rejects_unowned_file(monkeypatch, tmp_path):
    token_file = tmp_path / "token"
    token_file.write_text("secret")
    token_file.chmod(0o600)
    real_lstat = Path.lstat

    def fake_lstat(path):
        result = real_lstat(path)
        values = list(result)
        values[4] = os.getuid() + 1
        return os.stat_result(values)

    monkeypatch.setattr(Path, "lstat", fake_lstat)

    with pytest.raises(click.UsageError, match="current user"):
        ExplicitTokenFileAuthProvider(token_file).acquire("correlation")
