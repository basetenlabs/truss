import importlib
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

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


@pytest.mark.asyncio
async def test_trt_llm_truss_load_extension(trt_llm_truss_container_fs, helpers):
    app_path = trt_llm_truss_container_fs / "app"
    packages_path = trt_llm_truss_container_fs / "packages"
    with helpers.sys_paths(app_path, packages_path):
        model_wrapper_module = importlib.import_module("model_wrapper")
        model_wrapper_class = getattr(model_wrapper_module, "ModelWrapper")
        config = yaml.safe_load((app_path / "config.yaml").read_text())
        mock_extension = Mock()
        mock_extension.load = Mock()
        with patch.object(
            model_wrapper_module, "_load_extension", return_value=mock_extension
        ) as mock_load_extension:
            model_wrapper = model_wrapper_class(config)
            model_wrapper.load()
            called_with_specific_extension = any(
                call_args[0][0] == "trt_llm"
                for call_args in mock_load_extension.call_args_list
            )
            assert (
                called_with_specific_extension
            ), "Expected extension_name was not called"


@pytest.mark.asyncio
async def test_trt_llm_truss_predict(trt_llm_truss_container_fs, helpers):
    app_path = trt_llm_truss_container_fs / "app"
    packages_path = trt_llm_truss_container_fs / "packages"
    with helpers.sys_paths(app_path, packages_path), _change_directory(app_path):
        model_wrapper_module = importlib.import_module("model_wrapper")
        model_wrapper_class = getattr(model_wrapper_module, "ModelWrapper")
        config = yaml.safe_load((app_path / "config.yaml").read_text())

        expected_predict_response = "test"
        mock_predict_called = False

        async def mock_predict(return_value):
            nonlocal mock_predict_called
            mock_predict_called = True
            return expected_predict_response

        mock_engine = Mock(predict=mock_predict)
        mock_extension = Mock()
        mock_extension.load = Mock()
        mock_extension.model_args = Mock(return_value={"engine": mock_engine})
        with patch.object(
            model_wrapper_module, "_load_extension", return_value=mock_extension
        ):
            model_wrapper = model_wrapper_class(config)
            model_wrapper.load()
            resp = await model_wrapper.predict({})
            mock_extension.load.assert_called()
            mock_extension.model_args.assert_called()
            assert mock_predict_called
            assert resp == expected_predict_response


@pytest.mark.asyncio
async def test_trt_llm_truss_missing_model_py(trt_llm_truss_container_fs, helpers):
    app_path = trt_llm_truss_container_fs / "app"
    (app_path / "model" / "model.py").unlink()

    packages_path = trt_llm_truss_container_fs / "packages"
    with helpers.sys_paths(app_path, packages_path), _change_directory(app_path):
        model_wrapper_module = importlib.import_module("model_wrapper")
        model_wrapper_class = getattr(model_wrapper_module, "ModelWrapper")
        config = yaml.safe_load((app_path / "config.yaml").read_text())

        expected_predict_response = "test"
        mock_predict_called = False

        async def mock_predict(return_value):
            nonlocal mock_predict_called
            mock_predict_called = True
            return expected_predict_response

        mock_engine = Mock(predict=mock_predict)
        mock_extension = Mock()
        mock_extension.load = Mock()
        mock_extension.model_override = Mock(return_value=mock_engine)
        with patch.object(
            model_wrapper_module, "_load_extension", return_value=mock_extension
        ):
            model_wrapper = model_wrapper_class(config)
            model_wrapper.load()
            resp = await model_wrapper.predict({})
            mock_extension.load.assert_called()
            mock_extension.model_override.assert_called()
            assert mock_predict_called
            assert resp == expected_predict_response


@contextmanager
def _change_directory(new_directory: Path):
    original_directory = os.getcwd()
    os.chdir(str(new_directory))
    try:
        yield
    finally:
        os.chdir(original_directory)
