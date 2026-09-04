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

PINNED = "bdn://models/weights@b3:9232d924b49eeca26e1e665e8125af7a9f6a705090fa5a86247334743662a746"


def attachment_body(
    state: str = "READY",
    *,
    attachment_id: str = "vol_01ARZ3NDEKTSV4RRFFQ69G5FAV",
    include: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
) -> Dict[str, Any]:
    # Mirrors VolumeAttachment as hotloadd serializes it: include/exclude are
    # omitted when empty, pinned_source is always present.
    body: Dict[str, Any] = {
        "id": attachment_id,
        "revision": 2,
        "source": "bdn://models/weights:prod",
        "target": "weights",
        "path": "/bdn/mounts/weights",
        "pinned_source": PINNED,
        "state": state,
    }
    if include:
        body["include"] = include
    if exclude:
        body["exclude"] = exclude
    return body


def error_body(code: str, message: str, retryable: bool) -> Dict[str, Any]:
    return {"code": code, "message": message, "retryable": retryable}


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
        return

    def _respond(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        raw_body = self.rfile.read(length) if length else b""
        self.server.requests.append(
            {
                "method": self.command,
                "path": self.path,
                "headers": {key.lower(): value for key, value in self.headers.items()},
                "body": json.loads(raw_body) if raw_body else None,
            }
        )
        if not self.server.responses:
            raise AssertionError(f"unexpected request {self.command} {self.path}")
        response = self.server.responses.popleft()
        if response is DISCONNECT:
            self.close_connection = True
            self.wfile.close()
            return
        status, body = response
        encoded = json.dumps(body).encode("utf-8") if body is not None else b""
        self.send_response(status)
        if encoded:
            self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        if encoded:
            self.wfile.write(encoded)


@contextmanager
def hotload_server(
    socket_path: Path, responses: List[Any]
) -> Iterator[HotLoadTestServer]:
    server = HotLoadTestServer(socket_path, responses)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.fixture
def hotload_module():
    server_dir = Path(__file__).parents[3] / "templates" / "server"
    sys.path.insert(0, str(server_dir))
    try:
        for name in [m for m in sys.modules if m.startswith("bdn_hotload")]:
            del sys.modules[name]
        yield importlib.import_module("bdn_hotload")
    finally:
        sys.path.remove(str(server_dir))


def test_attach_generates_idempotency_key_and_returns_ready(
    tmp_path: Path, hotload_module
) -> None:
    socket_path = tmp_path / "hotload.sock"
    with hotload_server(socket_path, [(202, attachment_body())]) as server:
        client = hotload_module.HotLoadClient(socket_path)
        attachment = client.attach(
            "bdn://models/weights:prod", "weights", include=["*.safetensors"]
        )

    (request,) = server.requests
    assert request["method"] == "POST"
    assert request["path"] == "/v1/hotload/volumes"
    assert request["body"] == {
        "source": "bdn://models/weights:prod",
        "target": "weights",
        "include": ["*.safetensors"],
    }
    assert len(request["headers"]["idempotency-key"]) == 32
    assert attachment.state is hotload_module.AttachmentState.READY
    assert attachment.path == "/bdn/mounts/weights"
    assert attachment.pinned_source == PINNED
    assert attachment.include == ()
    assert attachment.id.startswith("vol_")


def test_default_socket_is_the_fixed_front_door(hotload_module) -> None:
    assert hotload_module.DEFAULT_SOCKET_PATH == Path("/bdn/hotload.sock")
    assert hotload_module.MOUNT_ROOT == Path("/bdn/mounts")


def test_list_get_and_detach_use_the_volumes_collection(
    tmp_path: Path, hotload_module
) -> None:
    socket_path = tmp_path / "hotload.sock"
    responses = [
        (200, {"volumes": [attachment_body(exclude=["tmp/**"])]}),
        (200, attachment_body()),
        (204, None),
    ]
    with hotload_server(socket_path, responses) as server:
        client = hotload_module.HotLoadClient(socket_path)
        volumes = client.list_volumes()
        fetched = client.get_volume("vol_01ARZ3NDEKTSV4RRFFQ69G5FAV")
        client.detach("vol_01ARZ3NDEKTSV4RRFFQ69G5FAV")

    assert [(r["method"], r["path"]) for r in server.requests] == [
        ("GET", "/v1/hotload/volumes"),
        ("GET", "/v1/hotload/volumes/vol_01ARZ3NDEKTSV4RRFFQ69G5FAV"),
        ("DELETE", "/v1/hotload/volumes/vol_01ARZ3NDEKTSV4RRFFQ69G5FAV"),
    ]
    assert volumes[0].exclude == ("tmp/**",)
    assert fetched.revision == 2


def test_attach_retries_a_dropped_connection_with_the_same_key(
    tmp_path: Path, hotload_module
) -> None:
    socket_path = tmp_path / "hotload.sock"
    with hotload_server(socket_path, [DISCONNECT, (202, attachment_body())]) as server:
        client = hotload_module.HotLoadClient(
            socket_path, retry_interval_sec=0, max_retries=1
        )
        client.attach("bdn://models/weights:prod", "weights", idempotency_key="k1")

    assert [r["headers"]["idempotency-key"] for r in server.requests] == ["k1", "k1"]


def test_api_error_exposes_the_stable_code(tmp_path: Path, hotload_module) -> None:
    socket_path = tmp_path / "hotload.sock"
    body = error_body(
        "TARGET_IN_USE", 'Hot Load target "weights" is already in use', False
    )
    with hotload_server(socket_path, [(409, body)]):
        client = hotload_module.HotLoadClient(socket_path)
        with pytest.raises(hotload_module.HotLoadAPIError) as raised:
            client.attach("bdn://models/weights:prod", "weights")

    assert raised.value.status_code == 409
    assert raised.value.code == "TARGET_IN_USE"
    assert raised.value.retryable is False


def test_retryable_api_error_is_retried_then_surfaced(
    tmp_path: Path, hotload_module
) -> None:
    socket_path = tmp_path / "hotload.sock"
    body = error_body("TOKEN_EXCHANGE_FAILED", "signer request failed", True)
    with hotload_server(socket_path, [(500, body), (500, body)]) as server:
        client = hotload_module.HotLoadClient(
            socket_path, retry_interval_sec=0, max_retries=1
        )
        with pytest.raises(hotload_module.HotLoadAPIError) as raised:
            client.attach("bdn://models/weights:prod", "weights")

    assert len(server.requests) == 2
    assert raised.value.code == "TOKEN_EXCHANGE_FAILED"


def test_failed_attachment_raises_attach_error(tmp_path: Path, hotload_module) -> None:
    socket_path = tmp_path / "hotload.sock"
    with hotload_server(socket_path, [(202, attachment_body("FAILED"))]):
        client = hotload_module.HotLoadClient(socket_path)
        with pytest.raises(hotload_module.HotLoadAttachError) as raised:
            client.attach("bdn://models/weights:prod", "weights")

    assert raised.value.attachment.state is hotload_module.AttachmentState.FAILED


def test_unknown_state_and_missing_fields_are_protocol_errors(
    tmp_path: Path, hotload_module
) -> None:
    socket_path = tmp_path / "hotload.sock"
    malformed = attachment_body()
    del malformed["pinned_source"]
    with hotload_server(
        socket_path, [(202, attachment_body("MOUNTING")), (202, malformed)]
    ):
        client = hotload_module.HotLoadClient(socket_path)
        with pytest.raises(hotload_module.HotLoadProtocolError):
            client.attach("bdn://models/weights:prod", "weights")
        with pytest.raises(hotload_module.HotLoadProtocolError):
            client.attach("bdn://models/weights:prod", "weights")


def test_missing_socket_is_connection_error(tmp_path: Path, hotload_module) -> None:
    client = hotload_module.HotLoadClient(
        tmp_path / "absent.sock", retry_interval_sec=0, max_retries=0
    )
    with pytest.raises(hotload_module.HotLoadConnectionError):
        client.list_volumes()


def test_attachment_id_must_be_a_single_path_segment(hotload_module) -> None:
    client = hotload_module.HotLoadClient("/nonexistent.sock")
    with pytest.raises(ValueError):
        client.get_volume("vol_1/../vol_2")
