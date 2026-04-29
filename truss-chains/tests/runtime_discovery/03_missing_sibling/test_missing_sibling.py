"""Adversarial: `get_service` raises with available-names list when the
requested chainlet is not registered in the chain."""

import json

import pytest

from truss_chains import public_types, runtime


@pytest.fixture
def dynamic_config_mount_dir(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "truss.templates.shared.dynamic_config_resolver.DYNAMIC_CONFIG_MOUNT_DIR",
        str(tmp_path),
    )
    yield


def test_missing_sibling_raises_with_available_names(
    tmp_path, dynamic_config_mount_dir
):
    config = {
        "Whisper": {"predict_url": "https://x.example/whisper"},
        "Diarizer": {"predict_url": "https://x.example/diarizer"},
    }
    with (tmp_path / "dynamic_chainlet_config").open("w") as f:
        f.write(json.dumps(config))

    with pytest.raises(public_types.MissingDependencyError) as excinfo:
        runtime.get_service("Typo")

    msg = str(excinfo.value)
    # The error must name what was attempted.
    assert "Typo" in msg
    # And the available alternatives — so users can debug typos quickly.
    assert "Whisper" in msg
    assert "Diarizer" in msg
