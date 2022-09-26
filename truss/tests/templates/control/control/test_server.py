import shutil
import sys
from pathlib import Path

import pytest

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
def app(tmp_path):
    inf_serv_home = tmp_path / "app"
    inf_serv_test_data_path = str(Path(__file__).parent / "test_data" / "app")
    shutil.copytree(inf_serv_test_data_path, str(inf_serv_home))
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
    try:
        resp = client.post("/patch", json=patch.to_dict())
    finally:
        client.post("/stop_inference_server")
    assert resp.status_code == 200
    with (
        app.config["inference_server_home"] / "model" / "model.py"
    ).open() as model_file:
        new_model_file_content = model_file.read()
    assert new_model_file_content == mock_model_file_content
