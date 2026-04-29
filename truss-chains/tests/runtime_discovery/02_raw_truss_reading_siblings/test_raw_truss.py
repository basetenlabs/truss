"""Drive the plain Truss's model.py with a fake dynamic_chainlet_config to
verify it discovers siblings via the public runtime API."""

import json
import pathlib
import sys

import pytest

# Make the plain_truss/model module importable.
PLAIN_TRUSS_MODEL_DIR = pathlib.Path(__file__).parent / "plain_truss" / "model"
sys.path.insert(0, str(PLAIN_TRUSS_MODEL_DIR))


@pytest.fixture
def dynamic_config_mount_dir(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "truss.templates.shared.dynamic_config_resolver.DYNAMIC_CONFIG_MOUNT_DIR",
        str(tmp_path),
    )
    yield


def _write_dynamic_config(tmp_path, payload):
    with (tmp_path / "dynamic_chainlet_config").open("w") as f:
        f.write(json.dumps(payload))


def test_plain_truss_picks_up_siblings(tmp_path, dynamic_config_mount_dir):
    _write_dynamic_config(
        tmp_path,
        {
            "Diarizer": {
                "predict_url": "https://chain-abc.api.baseten.co/.../diarizer/run_remote",
                "internal_url": {
                    "gateway_run_remote_url": "https://wp.api.baseten.co/.../diarizer/run_remote",
                    "hostname": "chain-abc.api.baseten.co",
                },
            }
        },
    )

    from model import Model  # noqa: import after monkeypatch

    m = Model()
    m.load()
    out = m.predict({})
    # internal_url wins (matches BasetenSession's selection rule).
    assert out["diarizer_url"] == "https://wp.api.baseten.co/.../diarizer/run_remote"
    assert out["auth_headers"] == {
        "Authorization": "Api-Key <from-secrets>",
        "Host": "chain-abc.api.baseten.co",
    }


def test_plain_truss_runs_standalone(tmp_path, dynamic_config_mount_dir):
    """No dynamic config file present — list_services returns empty,
    Model.load() short-circuits, predict returns the standalone shape."""
    from model import Model

    m = Model()
    m.load()
    out = m.predict({})
    assert out["diarizer_url"] is None
    assert out["auth_headers"] is None
