import importlib
import sys
import time
from pathlib import Path
from typing import Any

import pytest
import yaml


@pytest.fixture
def app_path(truss_container_fs: Path, helpers: Any):
    truss_container_app_path = truss_container_fs / "app"
    model_file_content = """
class Model:
    def __init__(self):
        self.load_count = 0
    def load(self):
        self.load_count += 1
        if self.load_count <= 2:
            raise RuntimeError('Simulated error')
    def predict(self, request):
        return request
    """
    with helpers.file_content(
        truss_container_app_path / "model" / "model.py",
        model_file_content,
    ), helpers.sys_path(truss_container_app_path):
        yield truss_container_app_path


# TODO: Make this test work
@pytest.mark.skip(
    reason="Succeeds when tests in this file are run alone, but fails with the whole suit"
)
def test_model_wrapper_load_error_once(app_path):
    if "model_wrapper" in sys.modules:
        model_wrapper_module = sys.modules["model_wrapper"]
        importlib.reload(model_wrapper_module)
    else:
        model_wrapper_module = importlib.import_module("model_wrapper")
    model_wraper_class = getattr(model_wrapper_module, "ModelWrapper")
    config = yaml.safe_load((app_path / "config.yaml").read_text())
    model_wrapper = model_wraper_class(config)
    model_wrapper.load()
    # Allow load thread to execute
    time.sleep(1)
    output = model_wrapper.predict({})
    assert output == {}
    assert model_wrapper._model.load_count == 3


# TODO: Make this test work
@pytest.mark.skip(
    reason="Succeeds when tests in this file are run alone, but fails with the whole suit"
)
def test_model_wrapper_load_error_more_than_allowed(app_path, helpers):
    with helpers.env_var("NUM_LOAD_RETRIES_TRUSS", "0"):
        if "model_wrapper" in sys.modules:
            model_wrapper_module = sys.modules["model_wrapper"]
            importlib.reload(model_wrapper_module)
        else:
            model_wrapper_module = importlib.import_module("model_wrapper")
        model_wraper_class = getattr(model_wrapper_module, "ModelWrapper")
        config = yaml.safe_load((app_path / "config.yaml").read_text())
        model_wrapper = model_wraper_class(config)
        model_wrapper.load()
        # Allow load thread to execute
        time.sleep(1)
        assert model_wrapper.load_failed()


@pytest.mark.integration
async def test_model_wrapper_streaming_timeout(app_path):
    if "model_wrapper" in sys.modules:
        model_wrapper_module = sys.modules["model_wrapper"]
        importlib.reload(model_wrapper_module)
    else:
        model_wrapper_module = importlib.import_module("model_wrapper")
    model_wraper_class = getattr(model_wrapper_module, "ModelWrapper")

    # Create an instance of ModelWrapper with streaming_read_timeout set to 5 seconds
    config = yaml.safe_load((app_path / "config.yaml").read_text())
    config["runtime"]["streaming_read_timeout"] = 5
    model_wrapper = model_wraper_class(config)
    model_wrapper.load()
    assert model_wrapper._config.get("runtime").get("streaming_read_timeout") == 5
