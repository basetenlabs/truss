import sys
from pathlib import Path

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
def app(tmp_path, truss_container_fs):
    inf_serv_home = truss_container_fs / "app"
    control_app = create_app(
        {
            "inference_server_home": inf_serv_home,
            "inference_server_process_args": ["python", "inference_server.py"],
            "control_server_host": "0.0.0.0",
            "control_server_port": 8081,
        }
    )
    yield control_app


@pytest.fixture()
def client(app):
    return app.test_client()


def test_restart_server(client):
    resp = client.post("/stop_inference_server")
    assert resp.status_code == 200
    assert "error" not in resp.json
    assert "msg" in resp.json

    # Try second restart
    resp = client.post("/stop_inference_server")
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
        resp = client.post("/patch", json=[patch.to_dict()])
    finally:
        client.post("/stop_inference_server")
    assert resp.status_code == 200
    assert "error" not in resp.json
    assert "msg" in resp.json
