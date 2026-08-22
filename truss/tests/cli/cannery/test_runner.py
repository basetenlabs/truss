import io
from pathlib import Path
from unittest.mock import Mock

import click
import pytest

from truss.cli.cannery import runner
from truss.cli.cannery.auth import ExchangedToken
from truss.cli.cannery.config import ActiveRemote, CanneryConfig
from truss.remote.truss_remote import RemoteConfig


class FakeProcess:
    def __init__(self):
        self.stdout = io.StringIO('{"protocol_version":1}')
        self.stderr = io.StringIO("")
        self.returncode = 0

    def wait(self, timeout=None):
        return self.returncode

    def poll(self):
        return self.returncode


def _remote_config() -> CanneryConfig:
    remote_config = RemoteConfig(
        name="baseten",
        configs={
            "remote_provider": "baseten",
            "remote_url": "https://app.baseten.co",
            "api_key": "baseten-secret",
        },
    )
    return CanneryConfig(
        api="https://bdn.baseten.co",
        org="org",
        active_remote=ActiveRemote(
            name="baseten",
            remote_url="https://app.baseten.co",
            config=remote_config,
        ),
    )


def test_remote_without_exchange_adapter_fails_before_subprocess(monkeypatch):
    popen = Mock()
    monkeypatch.setattr(runner.subprocess, "Popen", popen)

    with pytest.raises(click.UsageError, match="RUN-869"):
        runner.run_cannery(
            ["ls"],
            config_resolver=_remote_config,
            binary_resolver=lambda: "/bin/cannery",
        )

    popen.assert_not_called()


def test_exchange_token_uses_environment_and_is_always_cleaned_up(monkeypatch):
    adapter = Mock()
    adapter.exchange.return_value = ExchangedToken("cannery-secret")
    token_path = None
    child_correlation_id = None

    def start_process(argv, **kwargs):
        nonlocal token_path, child_correlation_id
        token_path = Path(kwargs["env"]["CANNERY_AUTH_TOKEN_FILE"])
        child_correlation_id = kwargs["env"]["CANNERY_CORRELATION_ID"]
        assert token_path.read_text() == "cannery-secret"
        assert "cannery-secret" not in argv
        assert str(token_path) not in argv
        return FakeProcess()

    monkeypatch.setattr(runner.subprocess, "Popen", start_process)

    runner.run_cannery(
        ["ls"],
        config_resolver=_remote_config,
        binary_resolver=lambda: "/bin/cannery",
        exchange_adapter=adapter,
    )

    assert token_path is not None
    assert not token_path.exists()
    assert adapter.exchange.call_args.args[1] == child_correlation_id


def test_exchange_token_is_cleaned_up_when_subprocess_start_fails(monkeypatch):
    adapter = Mock()
    adapter.exchange.return_value = ExchangedToken("cannery-secret")
    token_path = None

    def fail_process(argv, **kwargs):
        nonlocal token_path
        token_path = Path(kwargs["env"]["CANNERY_AUTH_TOKEN_FILE"])
        assert token_path.exists()
        raise OSError("cannot execute")

    monkeypatch.setattr(runner.subprocess, "Popen", fail_process)

    with pytest.raises(click.ClickException, match="Failed to start"):
        runner.run_cannery(
            ["ls"],
            config_resolver=_remote_config,
            binary_resolver=lambda: "/bin/cannery",
            exchange_adapter=adapter,
        )

    assert token_path is not None
    assert not token_path.exists()
