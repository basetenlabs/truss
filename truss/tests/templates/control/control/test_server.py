import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Dict

import pytest

# Needed to simulate the set up on the model docker container
sys.path.append(
    str(
        Path(__file__).parent.parent.parent.parent.parent
        / "templates"
        / "control"
        / "control"
    )
)

from truss.templates.control.control.application import create_app  # noqa
from truss.templates.control.control.helpers.types import (  # noqa
    Action,
    ModelCodePatch,
    Patch,
    PatchType,
)


@pytest.fixture
def truss_original_hash():
    return "1234"


@pytest.fixture
def app(truss_container_fs, truss_original_hash):
    with _env_var({"HASH_TRUSS": truss_original_hash}):
        inf_serv_home = truss_container_fs / "app"
        control_app = create_app(
            {
                "inference_server_home": inf_serv_home,
                "inference_server_process_args": ["python", "inference_server.py"],
                "control_server_host": "0.0.0.0",
                "control_server_port": 8081,
                "inference_server_port": 8082,
                "oversee_inference_server": False,
            }
        )
        yield control_app


@pytest.fixture()
def client(app):
    return app.test_client()


def test_restart_server(client):
    resp = client.post("/control/stop_inference_server")
    assert resp.status_code == 200
    assert "error" not in resp.json
    assert "msg" in resp.json

    # Try second restart
    resp = client.post("/control/stop_inference_server")
    assert resp.status_code == 200
    assert "error" not in resp.json
    assert "msg" in resp.json


def test_patch_model_code_update_existing(app, client):
    mock_model_file_content = """
class Model:
    def predict(self, request):
        return {'prediction': [1]}
"""
    patch = Patch(
        type=PatchType.MODEL_CODE,
        body=ModelCodePatch(
            action=Action.UPDATE,
            path="model.py",
            content=mock_model_file_content,
        ),
    )
    _apply_patch(client, patch)
    with (
        app.config["inference_server_home"] / "model" / "model.py"
    ).open() as model_file:
        new_model_file_content = model_file.read()
    assert new_model_file_content == mock_model_file_content


def test_patch_model_code_create_new(app, client):
    empty_content = ""
    patch = Patch(
        type=PatchType.MODEL_CODE,
        body=ModelCodePatch(
            action=Action.UPDATE,
            path="touched",
            content=empty_content,
        ),
    )
    _apply_patch(client, patch)
    assert (app.config["inference_server_home"] / "model" / "touched").exists()


def test_patch_model_code_create_in_new_dir(app, client):
    empty_content = ""
    patch = Patch(
        type=PatchType.MODEL_CODE,
        body=ModelCodePatch(
            action=Action.UPDATE,
            path="new_directory/touched",
            content=empty_content,
        ),
    )
    _apply_patch(client, patch)
    assert (
        app.config["inference_server_home"] / "model" / "new_directory" / "touched"
    ).exists()


def test_404(client):
    resp = client.post("/control/nonexitant")
    assert resp.status_code == 404


def test_invalid_patch(client):
    try:
        patch_request = {
            "hash": "dummy",
            "prev_hash": "invalid",
            "patches": [],
        }
        resp = client.post("/control/patch", json=patch_request)
    finally:
        client.post("/control/stop_inference_server")
    assert resp.status_code == 200
    assert "error" in resp.json
    assert "expected prev hash" in resp.json["error"]
    assert "msg" not in resp.json


def test_patch_model_code_delete(app, client):
    patch = Patch(
        type=PatchType.MODEL_CODE,
        body=ModelCodePatch(
            action=Action.REMOVE,
            path="dummy",
            content=None,
        ),
    )
    assert (app.config["inference_server_home"] / "model" / "dummy").exists()
    _apply_patch(client, patch)
    assert not (app.config["inference_server_home"] / "model" / "dummy").exists()


def _apply_patch(client, patch: Patch):
    try:
        original_hash = client.get("/control/truss_hash").json["result"]
        patch_request = {
            "hash": "dummy",
            "prev_hash": original_hash,
            "patches": [patch.to_dict()],
        }
        resp = client.post("/control/patch", json=patch_request)
    finally:
        client.post("/control/stop_inference_server")
    assert resp.status_code == 200
    assert "error" not in resp.json
    assert "msg" in resp.json


@contextmanager
def _env_var(kvs: Dict[str, str]):
    orig_env = os.environ.copy()
    try:
        os.environ.update(kvs)
        yield
    finally:
        os.environ.clear()
        os.environ.update(orig_env)
