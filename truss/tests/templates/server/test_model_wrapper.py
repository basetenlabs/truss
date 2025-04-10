import importlib
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import opentelemetry.sdk.trace as sdk_trace
import pytest
import yaml
from starlette.requests import Request


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def connected_request():
    mock_request = MagicMock(spec=Request)
    mock_request.is_disconnected = AsyncMock(return_value=False)
    return mock_request


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
    with (
        helpers.file_content(
            truss_container_app_path / "model" / "model.py", model_file_content
        ),
        helpers.sys_path(truss_container_app_path),
    ):
        yield truss_container_app_path


@pytest.mark.anyio
async def test_model_wrapper_load_error_once(app_path, connected_request):
    if "model_wrapper" in sys.modules:
        model_wrapper_module = sys.modules["model_wrapper"]
        importlib.reload(model_wrapper_module)
    else:
        model_wrapper_module = importlib.import_module("model_wrapper")
    model_wrapper_class = getattr(model_wrapper_module, "ModelWrapper")
    config = yaml.safe_load((app_path / "config.yaml").read_text())
    os.chdir(app_path)
    model_wrapper = model_wrapper_class(config, sdk_trace.NoOpTracer())
    model_wrapper.load()
    # Allow load thread to execute
    time.sleep(1)
    output = await model_wrapper.predict({}, connected_request)
    assert output == {}
    assert model_wrapper._model.load_count == 2


def test_model_wrapper_load_error_more_than_allowed(app_path, helpers):
    with helpers.env_var("NUM_LOAD_RETRIES_TRUSS", "0"):
        if "model_wrapper" in sys.modules:
            model_wrapper_module = sys.modules["model_wrapper"]
            importlib.reload(model_wrapper_module)
        else:
            model_wrapper_module = importlib.import_module("model_wrapper")
        model_wrapper_class = getattr(model_wrapper_module, "ModelWrapper")
        config = yaml.safe_load((app_path / "config.yaml").read_text())
        os.chdir(app_path)
        model_wrapper = model_wrapper_class(config, sdk_trace.NoOpTracer())
        model_wrapper.load()
        # Allow load thread to execute
        time.sleep(1)
        assert model_wrapper.load_failed


@pytest.mark.anyio
@pytest.mark.integration
async def test_model_wrapper_streaming_timeout(app_path):
    if "model_wrapper" in sys.modules:
        model_wrapper_module = sys.modules["model_wrapper"]
        importlib.reload(model_wrapper_module)
    else:
        model_wrapper_module = importlib.import_module("model_wrapper")
    model_wrapper_class = getattr(model_wrapper_module, "ModelWrapper")

    # Create an instance of ModelWrapper with streaming_read_timeout set to 5 seconds
    config = yaml.safe_load((app_path / "config.yaml").read_text())
    config["runtime"]["streaming_read_timeout"] = 5
    model_wrapper = model_wrapper_class(config, sdk_trace.NoOpTracer())
    model_wrapper.load()
    assert model_wrapper._config.get("runtime").get("streaming_read_timeout") == 5


@pytest.mark.anyio
async def test_trt_llm_truss_init_extension(trt_llm_truss_container_fs, helpers):
    app_path = trt_llm_truss_container_fs / "app"
    packages_path = trt_llm_truss_container_fs / "packages"
    with _clear_model_load_modules(), helpers.sys_paths(app_path, packages_path):
        model_wrapper_module = importlib.import_module("model_wrapper")
        model_wrapper_class = getattr(model_wrapper_module, "ModelWrapper")
        config = yaml.safe_load((app_path / "config.yaml").read_text())
        mock_extension = Mock()
        mock_extension.load = Mock()
        with patch.object(
            model_wrapper_module, "_init_extension", return_value=mock_extension
        ) as mock_init_extension:
            model_wrapper = model_wrapper_class(config, sdk_trace.NoOpTracer())
            model_wrapper.load()
            called_with_specific_extension = any(
                call_args[0][0] == "trt_llm"
                for call_args in mock_init_extension.call_args_list
            )
            assert called_with_specific_extension, (
                "Expected extension_name was not called"
            )


@pytest.mark.anyio
async def test_trt_llm_truss_predict(
    trt_llm_truss_container_fs, helpers, connected_request
):
    app_path = trt_llm_truss_container_fs / "app"
    packages_path = trt_llm_truss_container_fs / "packages"
    with (
        _clear_model_load_modules(),
        helpers.sys_paths(app_path, packages_path),
        _change_directory(app_path),
    ):
        model_wrapper_module = importlib.import_module("model_wrapper")
        model_wrapper_class = getattr(model_wrapper_module, "ModelWrapper")
        config = yaml.safe_load((app_path / "config.yaml").read_text())

        expected_predict_response = "test"
        mock_predict_called = False

        async def mock_predict(return_value, request):
            nonlocal mock_predict_called
            mock_predict_called = True
            return expected_predict_response

        mock_engine = Mock(predict=mock_predict)
        mock_extension = Mock()
        mock_extension.load = Mock()
        mock_extension.model_args = Mock(return_value={"engine": mock_engine})
        with patch.object(
            model_wrapper_module, "_init_extension", return_value=mock_extension
        ):
            model_wrapper = model_wrapper_class(config, sdk_trace.NoOpTracer())
            model_wrapper.load()
            resp = await model_wrapper.predict({}, connected_request)
            mock_extension.load.assert_called()
            mock_extension.model_args.assert_called()
            assert mock_predict_called
            assert resp == expected_predict_response


@pytest.mark.anyio
async def test_trt_llm_truss_missing_model_py(
    trt_llm_truss_container_fs, helpers, connected_request
):
    app_path = trt_llm_truss_container_fs / "app"
    (app_path / "model" / "model.py").unlink()

    packages_path = trt_llm_truss_container_fs / "packages"
    with (
        _clear_model_load_modules(),
        helpers.sys_paths(app_path, packages_path),
        _change_directory(app_path),
    ):
        model_wrapper_module = importlib.import_module("model_wrapper")
        model_wrapper_class = getattr(model_wrapper_module, "ModelWrapper")
        config = yaml.safe_load((app_path / "config.yaml").read_text())

        expected_predict_response = "test"
        mock_predict_called = False

        async def mock_predict(return_value, request: Request):
            nonlocal mock_predict_called
            mock_predict_called = True
            return expected_predict_response

        mock_engine = Mock(predict=mock_predict, spec=["predict"])
        mock_extension = Mock()
        mock_extension.load = Mock()
        mock_extension.model_override = Mock(return_value=mock_engine)
        with patch.object(
            model_wrapper_module, "_init_extension", return_value=mock_extension
        ):
            model_wrapper = model_wrapper_class(config, sdk_trace.NoOpTracer())
            model_wrapper.load()
            resp = await model_wrapper.predict({}, connected_request)
            mock_extension.load.assert_called()
            mock_extension.model_override.assert_called()
            assert mock_predict_called
            assert resp == expected_predict_response


@pytest.mark.anyio
async def test_open_ai_completion_endpoints(
    open_ai_container_fs, helpers, connected_request
):
    app_path = open_ai_container_fs / "app"
    with (
        _clear_model_load_modules(),
        helpers.sys_paths(app_path),
        _change_directory(app_path),
    ):
        model_wrapper_module = importlib.import_module("model_wrapper")
        model_wrapper_class = getattr(model_wrapper_module, "ModelWrapper")
        config = yaml.safe_load((app_path / "config.yaml").read_text())

        model_wrapper = model_wrapper_class(config, sdk_trace.NoOpTracer())
        model_wrapper.load()

        predict_resp = await model_wrapper.predict({}, connected_request)
        assert predict_resp == "predict"

        completions_resp = await model_wrapper.completions({}, connected_request)
        assert completions_resp == "completions"

        chat_completions_resp = await model_wrapper.chat_completions(
            {}, connected_request
        )
        assert chat_completions_resp == "chat_completions"


@contextmanager
def _change_directory(new_directory: Path):
    original_directory = os.getcwd()
    os.chdir(str(new_directory))
    try:
        yield
    finally:
        os.chdir(original_directory)


@contextmanager
def _clear_model_load_modules():
    """Clear dangling references to model and model_wrapper modules

    We do this before to clear any debris from before, and after to clean up
    after self. This is meant for cases where we simulate running a truss model
    in process, where these modules are loaded dyamically.
    """
    # These are left over by TrussModuleLoader used by local prediction tests.
    # TODO(pankaj) Find a way for TrussModuleLoader to clean up after itself.
    _remove_model_load_sys_modules()
    yield
    _remove_model_load_sys_modules()


def _remove_model_load_sys_modules():
    if "model" in sys.modules:
        del sys.modules["model"]
    if "model.model" in sys.modules:
        del sys.modules["model.model"]
    if "model_wrapper" in sys.modules:
        del sys.modules["model_wrapper"]
