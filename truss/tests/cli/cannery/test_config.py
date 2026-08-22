import click
import pytest

from truss.cli.cannery import config
from truss.remote.truss_remote import RemoteConfig


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    monkeypatch.delenv("TRUSS_CANNERY_API", raising=False)
    monkeypatch.delenv("TRUSS_CANNERY_ORG", raising=False)


def test_explicit_endpoint_override_bypasses_remote_resolution(monkeypatch):
    monkeypatch.setattr(
        config.RemoteFactory,
        "get_available_config_names",
        lambda: pytest.fail("remote resolution must be bypassed"),
    )
    monkeypatch.setenv("TRUSS_CANNERY_API", "http://localhost:9999")

    resolved = config.resolve_cannery_config()

    assert resolved.api == "http://localhost:9999"
    assert resolved.active_remote is None


def test_no_configured_remote_preserves_local_development_default(monkeypatch):
    monkeypatch.setattr(
        config.RemoteFactory, "get_available_config_names", lambda: []
    )

    resolved = config.resolve_cannery_config()

    assert resolved.api == config.DEFAULT_LOCAL_API
    assert resolved.org == config.DEFAULT_LOCAL_ORG


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
        config.RemoteFactory,
        "get_available_config_names",
        lambda: ["production"],
    )
    monkeypatch.setattr(
        config.remote_cli,
        "inquire_remote_name",
        lambda allow_create: "production",
    )
    monkeypatch.setattr(
        config.RemoteFactory,
        "load_remote_config",
        lambda remote_name: remote_config,
    )

    resolved = config.resolve_cannery_config()

    assert resolved.api == "https://bdn.baseten.co"
    assert resolved.active_remote is not None
    assert resolved.active_remote.name == "production"


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
        config.RemoteFactory,
        "load_remote_config",
        lambda remote_name: remote_config,
    )

    with pytest.raises(click.UsageError, match="No Cannery endpoint"):
        config.resolve_cannery_config()
