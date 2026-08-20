import importlib
import json
import socket
import socketserver
import sys
import threading
from collections import deque
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import pytest

if not hasattr(socket, "AF_UNIX") or not hasattr(socketserver, "UnixStreamServer"):
    pytest.skip(
        "Hot Load communicates through Unix-domain sockets", allow_module_level=True
    )


def mount_body(
    state: str,
    *,
    revision: Optional[int] = None,
    error: Optional[Dict[str, Any]] = None,
    include: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
) -> Dict[str, Any]:
    revisions = {
        "ACCEPTED": 1,
        "RESOLVING": 2,
        "MOUNTING": 3,
        "READY": 4,
        "FAILED": 4,
        "UNMOUNTING": 5,
    }
    return {
        "id": "mount_01",
        "revision": revision or revisions[state],
        "source": "bdn://models/weights@deadbeef",
        "target": "weights",
        "include": include or [],
        "exclude": exclude or [],
        "path": "/bdn/mounts/weights",
        "pinned_source": (
            "bdn://models/weights@deadbeef" if state == "READY" else None
        ),
        "state": state,
        "error": error,
    }


DISCONNECT = object()


class HotLoadTestServer(socketserver.UnixStreamServer):
    allow_reuse_address = True

    def __init__(self, socket_path: Path, responses: List[Any]) -> None:
        self.responses = deque(responses)
        self.requests: List[Dict[str, Any]] = []
        super().__init__(str(socket_path), HotLoadRequestHandler)


class HotLoadRequestHandler(BaseHTTPRequestHandler):
    server: HotLoadTestServer

    def do_GET(self) -> None:
        self._respond()

    def do_POST(self) -> None:
        self._respond()

    def do_DELETE(self) -> None:
        self._respond()

    def log_message(self, format: str, *args: Any) -> None:
        pass

    def _respond(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(length)
        self.server.requests.append(
            {
                "method": self.command,
                "path": self.path,
                "headers": dict(self.headers.items()),
                "body": json.loads(raw_body) if raw_body else None,
            }
        )
        response = self.server.responses.popleft()
        if response is DISCONNECT:
            self.connection.shutdown(socket.SHUT_RDWR)
            self.connection.close()
            return
        status, payload = response
        if isinstance(payload, bytes):
            body = payload
        elif payload is None:
            body = b""
        else:
            body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Content-Type", "application/json")
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)


@contextmanager
def run_server(tmp_path: Path, responses: List[Any]) -> Iterator[HotLoadTestServer]:
    server = HotLoadTestServer(tmp_path / "hotload.sock", responses)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.fixture
def hotload_module(truss_container_fs, helpers):
    app_path = truss_container_fs / "app"
    assert (app_path / "bdn_hotload" / "client.py").is_file()
    with helpers.sys_path(app_path):
        yield importlib.import_module("bdn_hotload")
    sys.modules.pop("bdn_hotload.client", None)
    sys.modules.pop("bdn_hotload", None)


def client(server: HotLoadTestServer, hotload_module):
    return hotload_module.HotLoadClient(
        server.server_address, poll_interval_sec=0, request_timeout_sec=1
    )


def test_mount_generates_idempotency_key_and_waits_until_ready(
    tmp_path: Path, hotload_module
) -> None:
    responses = [
        (202, mount_body("ACCEPTED")),
        (200, mount_body("RESOLVING")),
        (200, mount_body("MOUNTING")),
        (200, mount_body("READY")),
    ]
    with run_server(tmp_path, responses) as server:
        mount = client(server, hotload_module).mount(
            "bdn://models/weights@deadbeef", "weights"
        )

    assert mount.state == hotload_module.MountState.READY
    assert mount.path == "/bdn/mounts/weights"
    assert mount.pinned_source == "bdn://models/weights@deadbeef"
    assert server.requests[0]["method"] == "POST"
    assert server.requests[0]["path"] == "/v1/hotload"
    assert server.requests[0]["body"] == {
        "source": "bdn://models/weights@deadbeef",
        "target": "weights",
    }
    assert server.requests[0]["headers"]["Idempotency-Key"]
    assert [request["path"] for request in server.requests[1:]] == [
        "/v1/hotload/mount_01/watch?after=1&timeout_ms=30000",
        "/v1/hotload/mount_01/watch?after=2&timeout_ms=30000",
        "/v1/hotload/mount_01/watch?after=3&timeout_ms=30000",
    ]


def test_create_list_get_and_delete_resource_operations(
    tmp_path: Path, hotload_module
) -> None:
    accepted = mount_body("ACCEPTED")
    ready = mount_body("READY")
    responses = [(202, accepted), (200, {"mounts": [ready]}), (200, ready), (204, None)]
    with run_server(tmp_path, responses) as server:
        hotload = client(server, hotload_module)
        created = hotload.create_mount(
            "bdn://models/weights@deadbeef", "weights", idempotency_key="deploy-42"
        )
        mounts = hotload.list_mounts()
        fetched = hotload.get_mount(created.id)
        hotload.delete_mount(created.id)

    assert created.state == hotload_module.MountState.ACCEPTED
    assert mounts == [fetched]
    assert fetched.state == hotload_module.MountState.READY
    assert server.requests[0]["headers"]["Idempotency-Key"] == "deploy-42"
    assert [(request["method"], request["path"]) for request in server.requests] == [
        ("POST", "/v1/hotload"),
        ("GET", "/v1/hotload"),
        ("GET", "/v1/hotload/mount_01"),
        ("DELETE", "/v1/hotload/mount_01"),
    ]


def test_create_retries_with_the_same_idempotency_key(
    tmp_path: Path, hotload_module
) -> None:
    responses = [DISCONNECT, (202, mount_body("ACCEPTED"))]
    with run_server(tmp_path, responses) as server:
        hotload = hotload_module.HotLoadClient(
            server.server_address,
            poll_interval_sec=0,
            request_timeout_sec=1,
            max_retries=1,
        )
        hotload.create_mount("bdn://models/weights@deadbeef", "weights")

    assert len(server.requests) == 2
    assert (
        server.requests[0]["headers"]["Idempotency-Key"]
        == server.requests[1]["headers"]["Idempotency-Key"]
    )


def test_api_error_preserves_stable_error_fields(
    tmp_path: Path, hotload_module
) -> None:
    response = {
        "error": {
            "code": "NAMESPACE_DENIED",
            "message": "namespace private is not allowed",
            "retryable": False,
        }
    }
    with run_server(tmp_path, [(403, response)]) as server:
        with pytest.raises(hotload_module.HotLoadAPIError) as raised:
            client(server, hotload_module).create_mount(
                "bdn://private/weights@deadbeef", "weights"
            )

    assert raised.value.status_code == 403
    assert raised.value.code == "NAMESPACE_DENIED"
    assert raised.value.message == "namespace private is not allowed"
    assert raised.value.retryable is False


def test_failed_mount_raises_mount_error(tmp_path: Path, hotload_module) -> None:
    failure = {
        "code": "MOUNT_TIMEOUT",
        "message": "Chowder did not become ready",
        "retryable": True,
    }
    responses = [
        (202, mount_body("ACCEPTED")),
        (200, mount_body("FAILED", error=failure)),
    ]
    with run_server(tmp_path, responses) as server:
        with pytest.raises(hotload_module.HotLoadMountError) as raised:
            client(server, hotload_module).mount(
                "bdn://models/weights@deadbeef", "weights"
            )

    assert raised.value.mount.state == hotload_module.MountState.FAILED
    assert raised.value.code == "MOUNT_TIMEOUT"
    assert raised.value.retryable is True


def test_inspect_partial_mount_watch_and_unmount_lifecycle(
    tmp_path: Path, hotload_module
) -> None:
    include = ["model/**", "tokenizer.json"]
    exclude = ["model/private/**"]
    inspection = {
        "source": "bdn://models/weights:prod",
        "pinned_source": "bdn://models/weights@deadbeef",
        "file_count": 3,
        "total_size": 42,
    }
    not_found = {
        "error": {
            "code": "MOUNT_NOT_FOUND",
            "message": "mount does not exist",
            "retryable": False,
        }
    }
    responses = [
        (200, inspection),
        (202, mount_body("ACCEPTED", include=include, exclude=exclude)),
        (200, mount_body("READY", include=include, exclude=exclude)),
        (204, None),
        (404, not_found),
    ]
    with run_server(tmp_path, responses) as server:
        hotload = client(server, hotload_module)
        inspected = hotload.inspect_source("bdn://models/weights:prod")
        mounted = hotload.mount(
            "bdn://models/weights@deadbeef", "weights", include=include, exclude=exclude
        )
        watched = list(hotload.watch(mounted.id, initial_mount=mounted))
        hotload.unmount(mounted.id)

    assert inspected.pinned_source == "bdn://models/weights@deadbeef"
    assert inspected.file_count == 3
    assert mounted.include == tuple(include)
    assert mounted.exclude == tuple(exclude)
    assert watched == [mounted]
    assert server.requests[0]["path"] == "/v1/hotload/inspect"
    assert server.requests[1]["body"] == {
        "source": "bdn://models/weights@deadbeef",
        "target": "weights",
        "include": include,
        "exclude": exclude,
    }
    assert server.requests[2]["path"] == (
        "/v1/hotload/mount_01/watch?after=1&timeout_ms=30000"
    )
    assert server.requests[3]["method"] == "DELETE"
    assert server.requests[4]["method"] == "GET"


def test_missing_socket_is_connection_error(tmp_path: Path, hotload_module) -> None:
    hotload = hotload_module.HotLoadClient(
        tmp_path / "missing.sock",
        poll_interval_sec=0,
        request_timeout_sec=0.1,
        max_retries=0,
    )

    with pytest.raises(hotload_module.HotLoadConnectionError, match="missing.sock"):
        hotload.list_mounts()
