import asyncio
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
from threading import Event
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import opentelemetry.sdk.trace as sdk_trace
import pytest
import yaml
from starlette.datastructures import Headers
from starlette.requests import Request
from starlette.responses import Response

from truss.templates.shared import serialization


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
    """Clear truss_server and model modules for clean import."""
    for mod in ["truss_server"] + [
        k for k in sys.modules if k == "model" or k.startswith("model.")
    ]:
        sys.modules.pop(mod, None)
    try:
        yield
    finally:
        for mod in ["truss_server"] + [
            k for k in sys.modules if k == "model" or k.startswith("model.")
        ]:
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
async def test_execute_request_installs_disconnect_watcher(app_path):
    with _clear_truss_server_modules(), _change_directory(app_path):
        truss_server_module = importlib.import_module("truss_server")
        model = MagicMock(skip_input_parsing=True)
        endpoints = truss_server_module.BasetenEndpoints(model, sdk_trace.NoOpTracer())
        endpoints.check_healthy = MagicMock()
        request = Request({"type": "http", "method": "POST", "headers": []})

        async def predict(inputs, model_request):
            assert inputs is None
            assert callable(model_request.state.watch_disconnect)
            return Response()

        response = await endpoints._execute_request(predict, request, b"")

    assert response.status_code == 200


@pytest.mark.anyio
async def test_disconnect_watcher_sets_event_from_worker_thread(app_path):
    with _clear_truss_server_modules(), _change_directory(app_path):
        truss_server_module = importlib.import_module("truss_server")
        receive_called = asyncio.Event()

        async def receive():
            receive_called.set()
            return {"type": "http.disconnect"}

        request = Request(
            {"type": "http", "method": "POST", "headers": []}, receive=receive
        )
        truss_server_module._install_disconnect_watcher(request)

        assert not receive_called.is_set()

        def stream_until_disconnected():
            with request.state.watch_disconnect() as disconnected:
                assert isinstance(disconnected, Event)
                yield b"first"
                assert disconnected.wait(timeout=1)

        stream = stream_until_disconnected()
        assert await asyncio.to_thread(next, stream) == b"first"
        await asyncio.wait_for(receive_called.wait(), timeout=1)

        def finish_stream():
            try:
                next(stream)
            except StopIteration:
                return True
            return False

        assert await asyncio.to_thread(finish_stream)
        assert receive_called.is_set()


@pytest.mark.anyio
async def test_disconnect_watcher_stops_when_context_exits(app_path):
    with _clear_truss_server_modules(), _change_directory(app_path):
        truss_server_module = importlib.import_module("truss_server")
        request = Request({"type": "http", "method": "POST", "headers": []})
        poll_started = asyncio.Event()
        poll_cancelled = asyncio.Event()

        async def wait_for_disconnect():
            poll_started.set()
            try:
                await asyncio.Future()
            finally:
                poll_cancelled.set()

        request.is_disconnected = AsyncMock(side_effect=wait_for_disconnect)
        truss_server_module._install_disconnect_watcher(request)

        with request.state.watch_disconnect() as disconnected:
            await asyncio.wait_for(poll_started.wait(), timeout=1)
            assert not disconnected.is_set()

        await asyncio.wait_for(poll_cancelled.wait(), timeout=1)
        assert not disconnected.is_set()


@pytest.mark.anyio
async def test_execute_request_sets_request_id_in_context(app_path):
    """Verify _execute_request sets request_id from x-baseten-request-id header in context."""
    request_id = "test-request-id-12345"
    mock_request = _make_connected_request(request_id)

    with (
        _clear_truss_server_modules(),
        _change_directory(app_path),
        patch("_truss_shared.log_config.request_id_context") as mock_request_id_context,
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
        patch("_truss_shared.log_config.request_id_context") as mock_request_id_context,
    ):
        endpoints = _get_endpoints(app_path)

        await endpoints.predict(
            model_name="model", request=mock_request, body_raw=b"{}"
        )

        mock_request_id_context.set.assert_called_once_with(None)


@pytest.mark.anyio
@pytest.mark.parametrize(
    "content_type",
    ["application/octet-stream; charset=binary", "Application/Octet-Stream"],
)
async def test_binary_request_accepts_content_type_variants(app_path, content_type):
    mock_request = _make_connected_request()
    mock_request.headers = Headers({"Content-Type": content_type})
    body = serialization.truss_msgpack_serialize({})

    with _clear_truss_server_modules(), _change_directory(app_path):
        endpoints = _get_endpoints(app_path)
        response = await endpoints.predict(
            model_name="model", request=mock_request, body_raw=body
        )

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/octet-stream"
    assert serialization.truss_msgpack_deserialize(response.body) == {"predictions": []}


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
        patch("_truss_shared.log_config.request_id_context") as mock_request_id_context,
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
        daemon=True,  # Don't block pytest exit if an assertion below fails.
    )
    subproc.start()
    proc_id = subproc.pid
    # The server should come up; poll instead of a fixed sleep, startup can
    # be slow on CI runners.
    assert _wait_for(lambda: _is_server_listening(port)), "server never started"
    os.kill(proc_id, signal.SIGTERM)
    subproc.join(timeout=30)
    # Print on purpose for help with debugging, otherwise hard to know what's going on
    print(Path(stdout_capture_file.name).read_text())
    assert not subproc.is_alive()
    assert _wait_for(lambda: not _is_server_listening(port)), "port still in use"


@pytest.mark.integration
def test_sync_generator_observes_tcp_disconnect(truss_container_fs, unused_tcp_port):
    model_file = truss_container_fs / "app" / "model" / "model.py"
    model_file.write_text(
        """\
from typing import Any

from fastapi import Request


class Model:
    def __init__(self, **kwargs) -> None:
        pass

    def load(self) -> None:
        pass

    def predict(self, model_input: Any, request: Request):
        with request.state.watch_disconnect() as disconnected:
            yield b"first"
            if disconnected.wait(timeout=5):
                print("SYNC_DISCONNECT_OBSERVED", flush=True)
            else:
                print("SYNC_DISCONNECT_MISSED", flush=True)
"""
    )

    stdout_capture_file = tempfile.NamedTemporaryFile()
    subproc = Process(
        target=_start_truss_server,
        args=(stdout_capture_file.name, truss_container_fs, unused_tcp_port),
        daemon=True,
    )
    subproc.start()

    model_url = f"http://127.0.0.1:{unused_tcp_port}/v1/models/x-4:predict"
    ready_url = f"http://127.0.0.1:{unused_tcp_port}/v1/models/x-4"

    def model_is_ready():
        try:
            return httpx.get(ready_url, timeout=1).status_code == 200
        except httpx.HTTPError:
            return False

    try:
        assert _wait_for(model_is_ready), "model server never became ready"

        with httpx.Client(timeout=5) as client:
            with client.stream("POST", model_url, json={}) as response:
                response.raise_for_status()
                assert next(response.iter_bytes()) == b"first"

        logs_path = Path(stdout_capture_file.name)
        assert _wait_for(
            lambda: "SYNC_DISCONNECT_OBSERVED" in logs_path.read_text(),
            timeout_sec=8,
            poll_sec=0.1,
        ), "synchronous generator did not observe the TCP disconnect"
        assert "SYNC_DISCONNECT_MISSED" not in logs_path.read_text()
    finally:
        if subproc.is_alive():
            os.kill(subproc.pid, signal.SIGTERM)
        subproc.join(timeout=30)
        print(Path(stdout_capture_file.name).read_text())

    assert not subproc.is_alive()
    assert _wait_for(lambda: not _is_server_listening(unused_tcp_port)), (
        "port still in use"
    )
    subproc.close()


@pytest.mark.anyio
async def test_hot_reload_with_namespace_subpackage(app_path):
    # Regression test for https://github.com/basetenlabs/truss/issues/2320:
    # When model/ is a regular package (__init__.py) but a subdirectory is a
    # namespace package (no __init__.py), hot reload evicts the parent from
    # sys.modules but leaves the namespace child. The child's _NamespacePath
    # then raises KeyError looking up the evicted parent.
    model_dir = app_path / "model"
    model_file = model_dir / "model.py"

    # analyzer/ is a namespace package (no __init__.py)
    analyzer_dir = model_dir / "analyzer"
    analyzer_dir.mkdir()
    request_dir = analyzer_dir / "request"
    request_dir.mkdir()
    (request_dir / "__init__.py").write_text("")
    (request_dir / "schema.py").write_text('class RequestModel:\n    name = "v1"\n')

    model_file.write_text("""\
from model.analyzer.request.schema import RequestModel

class Model:
    def predict(self, request):
        return {"version": RequestModel.name}
""")

    with _clear_truss_server_modules(), _change_directory(app_path):
        endpoints = _get_endpoints(app_path)
        mock_request = _make_connected_request()

        async def predict():
            resp = await endpoints.predict(
                model_name="model", request=mock_request, body_raw=b"{}"
            )
            assert resp.status_code == 200, f"predict failed: {resp.body}"
            return json.loads(resp.body)

        result = await predict()
        assert result["version"] == "v1"

        # Update and hot reload
        (request_dir / "schema.py").write_text('class RequestModel:\n    name = "v2"\n')
        resp = endpoints.hot_reload(mock_request)
        assert resp == {"msg": "Hot reload complete"}

        result = await predict()
        assert result["version"] == "v2"


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
            assert resp.status_code == 200, f"predict failed: {resp.body}"
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


@contextmanager
def _inference_server_app(app_path):
    """Yield a FastAPI app for the inference server with Prometheus mocked out.

    The Prometheus global registry is a process-level singleton.  Calling
    ``create_application()`` more than once per process would fail because the
    collectors would already have been unregistered.  We mock the registry and
    the metrics helpers so that routing (the thing under test here) is
    exercised without touching process-global state.
    """
    truss_server_module = importlib.import_module("truss_server")
    server = truss_server_module.TrussServer(
        http_port=8080,
        config_or_path=app_path / "config.yaml",
    )
    # on_startup launches background threads/tasks that are not needed for
    # routing tests and would outlive the test.
    with (
        patch.object(server, "on_startup"),
        patch.object(truss_server_module, "REGISTRY", MagicMock()),
        patch.object(truss_server_module, "make_asgi_app", return_value=MagicMock()),
        patch.object(truss_server_module, "metrics", MagicMock()),
    ):
        yield server.create_application()


@pytest.mark.anyio
async def test_404_for_unknown_path(app_path):
    """Non-existent paths must return 404, not 200."""
    with _clear_truss_server_modules(), _change_directory(app_path):
        with _inference_server_app(app_path) as app:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                for path in [
                    "/v1/nonexistent",
                    "/v1/models/model/nonexistent-subpath",
                    "/some-random-path",
                ]:
                    resp = await client.get(path)
                    assert resp.status_code == 404, (
                        f"GET {path}: expected 404, got {resp.status_code}"
                    )
                    resp = await client.post(path, content=b"{}")
                    assert resp.status_code == 404, (
                        f"POST {path}: expected 404, got {resp.status_code}"
                    )


@pytest.mark.anyio
async def test_404_for_unimplemented_optional_endpoint(app_path):
    """Optional endpoints (chat_completions, embeddings, etc.) raise
    ModelMethodNotImplemented -- which maps to HTTP 404 via the registered
    exception handler -- when the deployed model does not implement them."""
    with _clear_truss_server_modules(), _change_directory(app_path):
        truss_server_module = importlib.import_module("truss_server")
        # _get_endpoints loads the model so that check_healthy() passes.
        endpoints = _get_endpoints(app_path)
        errors_mod = truss_server_module.errors
        mock_request = _make_connected_request()

        for method_name in ("chat_completions", "completions", "embeddings"):
            endpoint_fn = getattr(endpoints, method_name)
            with pytest.raises(errors_mod.ModelMethodNotImplemented):
                await endpoint_fn(mock_request, b"{}")


def _is_server_listening(port):
    # Connect-based check: only an active listener counts, so lingering
    # TIME_WAIT sockets from a just-terminated server don't flake the test.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1.0)
        return s.connect_ex(("localhost", port)) == 0


def _wait_for(predicate, timeout_sec=30.0, poll_sec=0.5):
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(poll_sec)
    return predicate()
