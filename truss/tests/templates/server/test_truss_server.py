import importlib
import json
import os
import signal
import socket
import sys
import tempfile
import time
from contextlib import contextmanager
from multiprocessing import Process
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import opentelemetry.sdk.trace as sdk_trace
import pytest
import yaml


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def app_path(truss_container_fs, helpers):
    truss_container_app_path = truss_container_fs / "app"
    with helpers.sys_path(truss_container_app_path):
        yield truss_container_app_path


@contextmanager
def _change_directory(new_directory: Path):
    original_directory = os.getcwd()
    os.chdir(str(new_directory))
    try:
        yield
    finally:
        os.chdir(original_directory)


@contextmanager
def _clear_truss_server_modules():
    """Clear truss_server module for clean import."""
    modules_to_remove = ["truss_server"]
    for mod in modules_to_remove:
        sys.modules.pop(mod, None)
    yield
    for mod in modules_to_remove:
        sys.modules.pop(mod, None)


def _get_endpoints(app_path):
    """Create BasetenEndpoints from container app path."""
    model_wrapper_module = importlib.import_module("model_wrapper")
    truss_server_module = importlib.import_module("truss_server")
    config = yaml.safe_load((app_path / "config.yaml").read_text())
    model_wrapper = model_wrapper_module.ModelWrapper(config, sdk_trace.NoOpTracer())
    model_wrapper.load()
    time.sleep(1)  # Allow load thread to complete

    tracer = truss_server_module.tracing.get_truss_tracer(
        truss_server_module.SecretsResolver.get_secrets(config), config
    )
    return truss_server_module.BasetenEndpoints(model_wrapper, tracer)


def _make_connected_request(request_id=None):
    """Create a mock Request with headers and is_disconnected for predict flow."""
    mock_request = MagicMock()
    mock_request.headers.get = lambda key, default=None: (
        request_id if key == "x-baseten-request-id" else default
    )
    mock_request.is_disconnected = AsyncMock(return_value=False)
    return mock_request


@pytest.mark.anyio
async def test_execute_request_sets_request_id_in_context(app_path):
    """Verify _execute_request sets request_id from x-baseten-request-id header in context."""
    request_id = "test-request-id-12345"
    mock_request = _make_connected_request(request_id)

    with (
        _clear_truss_server_modules(),
        _change_directory(app_path),
        patch("shared.log_config.request_id_context") as mock_request_id_context,
    ):
        endpoints = _get_endpoints(app_path)

        await endpoints.predict(
            model_name="model", request=mock_request, body_raw=b"{}"
        )

        mock_request_id_context.set.assert_called_once_with(request_id)


@pytest.mark.anyio
async def test_execute_request_sets_none_when_no_request_id_header(app_path):
    """Verify _execute_request sets None in context when x-baseten-request-id is missing."""
    mock_request = _make_connected_request()

    with (
        _clear_truss_server_modules(),
        _change_directory(app_path),
        patch("shared.log_config.request_id_context") as mock_request_id_context,
    ):
        endpoints = _get_endpoints(app_path)

        await endpoints.predict(
            model_name="model", request=mock_request, body_raw=b"{}"
        )

        mock_request_id_context.set.assert_called_once_with(None)


@pytest.mark.anyio
async def test_websocket_sets_request_id_in_context(app_path):
    """Verify websocket sets request_id from x-baseten-request-id header in context."""
    request_id = "ws-request-id-67890"
    mock_ws = MagicMock()
    mock_ws.headers.get = lambda key, default=None: (
        request_id if key == "x-baseten-request-id" else default
    )
    mock_ws.accept = AsyncMock()
    mock_ws.close = AsyncMock()

    with (
        _clear_truss_server_modules(),
        _change_directory(app_path),
        patch("shared.log_config.request_id_context") as mock_request_id_context,
    ):
        endpoints = _get_endpoints(app_path)

        await endpoints.websocket(mock_ws)

        mock_request_id_context.set.assert_called_once_with(request_id)


def _start_truss_server(
    stdout_capture_file_path: str, truss_container_fs: Path, port: int
):
    """Module-level function to avoid pickling issues with multiprocessing."""
    sys.stdout = open(stdout_capture_file_path, "w")
    app_path = truss_container_fs / "app"
    sys.path.append(str(app_path))
    os.chdir(app_path)

    from truss_server import TrussServer

    server = TrussServer(http_port=port, config_or_path=app_path / "config.yaml")
    server.start()


@pytest.mark.integration
def test_truss_server_termination(truss_container_fs):
    port = 10123

    stdout_capture_file = tempfile.NamedTemporaryFile()
    subproc = Process(
        target=_start_truss_server,
        args=(stdout_capture_file.name, truss_container_fs, port),
    )
    subproc.start()
    proc_id = subproc.pid
    time.sleep(2.0)
    # Port should have been taken up by truss server
    assert not _is_port_available(port)
    os.kill(proc_id, signal.SIGTERM)
    time.sleep(2.0)
    # Print on purpose for help with debugging, otherwise hard to know what's going on
    print(Path(stdout_capture_file.name).read_text())
    assert not subproc.is_alive()
    # Port should be free now
    assert _is_port_available(port)


@pytest.mark.anyio
async def test_hot_reload_endpoint(app_path):
    # Tests the HTTP endpoint layer: successful hot reload changes predict
    # output, and a syntax error returns 422 while old predict still works.
    model_file = app_path / "model" / "model.py"

    original_model = """\
class Model:
    def __init__(self, **kwargs):
        self.load_count = 0
        self.predict_count = 0

    def load(self):
        self.load_count += 1

    def predict(self, request):
        self.predict_count += 1
        return {"version": "v1", "load_count": self.load_count, "predict_count": self.predict_count}
"""
    model_file.write_text(original_model)

    with _clear_truss_server_modules(), _change_directory(app_path):
        endpoints = _get_endpoints(app_path)
        mock_request = _make_connected_request()

        async def predict():
            resp = await endpoints.predict(
                model_name="model", request=mock_request, body_raw=b"{}"
            )
            return json.loads(resp.body)

        result = await predict()
        assert result["version"] == "v1"
        assert result["load_count"] == 1
        assert result["predict_count"] == 1

        # Hot reload with new code
        model_file.write_text("""\
class Model:
    def predict(self, request):
        self.predict_count += 1
        return {"version": "v2", "load_count": self.load_count, "predict_count": self.predict_count}
""")
        resp = endpoints.hot_reload(mock_request)
        assert resp == {"msg": "Hot reload complete"}

        result = await predict()
        assert result["version"] == "v2"
        assert result["load_count"] == 1
        assert result["predict_count"] == 2

        # Hot reload with syntax error (missing colon) returns 422
        model_file.write_text("class Model:\n    def predict(self, request)\n")
        resp = endpoints.hot_reload(mock_request)
        assert resp.status_code == 422
        assert "SyntaxError" in resp.body.decode()

        # Old predict still works with preserved state
        result = await predict()
        assert result["version"] == "v2"
        assert result["predict_count"] == 3


def _is_port_available(port):
    try:
        # Try to bind to the given port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("localhost", port))
            return True
    except socket.error:
        # Port is already in use
        return False
