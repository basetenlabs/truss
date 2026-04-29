"""Adversarial: behavior when no chain context is present."""

import pytest

from truss_chains import public_types, runtime


@pytest.fixture
def dynamic_config_mount_dir(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "truss.templates.shared.dynamic_config_resolver.DYNAMIC_CONFIG_MOUNT_DIR",
        str(tmp_path),
    )
    yield


def test_list_services_empty_when_no_context(tmp_path, dynamic_config_mount_dir):
    """No file written; list_services returns empty mapping (does NOT raise)."""
    assert runtime.list_services() == {}


def test_get_service_raises_clear_message(tmp_path, dynamic_config_mount_dir):
    with pytest.raises(public_types.MissingDependencyError) as excinfo:
        runtime.get_service("Anything")
    assert "not running inside a chain context" in str(excinfo.value)
