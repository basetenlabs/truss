import click
import pytest

from truss.cli.cannery import config
from truss.remote.truss_remote import RemoteConfig


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    monkeypatch.delenv("TRUSS_CANNERY_API", raising=False)
    monkeypatch.delenv("TRUSS_CANNERY_ORG", raising=False)
    monkeypatch.delenv("TRUSS_CANNERY_BIN", raising=False)
    monkeypatch.delenv("TRUSS_CANNERY_ALLOW_PATH", raising=False)


def test_explicit_endpoint_override_bypasses_remote_resolution(monkeypatch):
    monkeypatch.setattr(
        config.RemoteFactory,
        "get_available_config_names",
        lambda: pytest.fail("remote resolution must be bypassed"),
    )
    monkeypatch.setenv("TRUSS_CANNERY_API", "http://localhost:9999")
    monkeypatch.setenv("TRUSS_CANNERY_BIN", "/bin/cannery")

    resolved = config.resolve_cannery_config()

    assert resolved.api == "http://localhost:9999"
    assert resolved.active_remote is None


def test_no_configured_remote_fails_closed(monkeypatch):
    monkeypatch.setattr(config.RemoteFactory, "get_available_config_names", lambda: [])

    with pytest.raises(click.UsageError, match="truss auth login"):
        config.resolve_cannery_config()


def test_endpoint_is_derived_from_active_truss_remote(monkeypatch):
    remote_config = RemoteConfig(
        name="production",
        configs={
            "remote_provider": "baseten",
            "remote_url": "https://app.baseten.co/",
            "api_key": "not-observable",
        },
    )
    monkeypatch.setattr(
        config.RemoteFactory, "get_available_config_names", lambda: ["production"]
    )
    monkeypatch.setattr(
        config.remote_cli, "inquire_remote_name", lambda allow_create: "production"
    )
    monkeypatch.setattr(
        config.RemoteFactory, "load_remote_config", lambda remote_name: remote_config
    )

    resolved = config.resolve_cannery_config("production")

    assert resolved.api == "https://bdn.baseten.co"
    assert resolved.active_remote is not None
    assert resolved.active_remote.name == "production"


def test_explicit_remote_selects_one_of_multiple_configs(monkeypatch):
    remote_config = RemoteConfig(
        name="staging",
        configs={
            "remote_provider": "baseten",
            "remote_url": "https://app.baseten.co",
            "api_key": "not-observable",
        },
    )
    monkeypatch.setattr(
        config.RemoteFactory,
        "get_available_config_names",
        lambda: ["production", "staging"],
    )
    loaded_names = []
    monkeypatch.setattr(
        config.RemoteFactory,
        "load_remote_config",
        lambda remote_name: loaded_names.append(remote_name) or remote_config,
    )

    resolved = config.resolve_cannery_config("staging")

    assert resolved.active_remote is not None
    assert resolved.active_remote.name == "staging"
    assert loaded_names == ["staging"]


def test_multiple_remotes_require_explicit_selection(monkeypatch):
    monkeypatch.setattr(
        config.RemoteFactory,
        "get_available_config_names",
        lambda: ["production", "staging"],
    )
    monkeypatch.setattr(
        config.remote_cli,
        "inquire_remote_name",
        lambda allow_create: (_ for _ in ()).throw(
            click.UsageError(
                "Multiple remotes available. Please specify one with --remote."
            )
        ),
    )

    with pytest.raises(click.UsageError, match="specify one with --remote"):
        config.resolve_cannery_config()


def test_unknown_remote_endpoint_fails_closed(monkeypatch):
    remote_config = RemoteConfig(
        name="custom",
        configs={
            "remote_provider": "baseten",
            "remote_url": "https://custom.example.com",
        },
    )
    monkeypatch.setattr(
        config.RemoteFactory, "get_available_config_names", lambda: ["custom"]
    )
    monkeypatch.setattr(
        config.remote_cli, "inquire_remote_name", lambda allow_create: "custom"
    )
    monkeypatch.setattr(
        config.RemoteFactory, "load_remote_config", lambda remote_name: remote_config
    )

    with pytest.raises(click.UsageError, match="No public volume API endpoint"):
        config.resolve_cannery_config("custom")


def test_local_path_lookup_requires_explicit_opt_in(monkeypatch):
    monkeypatch.setattr(
        config.RemoteFactory, "get_available_config_names", lambda: pytest.fail()
    )
    monkeypatch.setenv("TRUSS_CANNERY_API", "http://127.0.0.1:8787")

    with pytest.raises(click.UsageError, match="TRUSS_CANNERY_BIN"):
        config.resolve_cannery_config()

    monkeypatch.setenv("TRUSS_CANNERY_ALLOW_PATH", "1")
    resolved = config.resolve_cannery_config()

    assert resolved.allow_path_fallback
