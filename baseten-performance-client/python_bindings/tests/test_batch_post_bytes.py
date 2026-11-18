"""batch_post / async_batch_post accept pre-encoded JSON bytes and send them byte for byte."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from baseten_performance_client import PerformanceClient


class _EchoHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        out = json.dumps(
            {"raw": body.decode(), "content_type": self.headers["Content-Type"]}
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *args):
        pass


@pytest.fixture(scope="module")
def client():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _EchoHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield PerformanceClient(
        base_url=f"http://127.0.0.1:{server.server_port}", api_key="test-key"
    )
    server.shutdown()


# Odd spacing, key order and surrounding whitespace prove the bytes are sent unchanged.
RAW = b' \n{"b":  [1.5, 2e-3],"a":"\xc3\xa9"}\n'
PAYLOADS = [RAW, {"x": 1}]


def _check(response):
    assert [d["raw"] for d in response.data] == [RAW.decode(), '{"x":1}']
    assert all(d["content_type"] == "application/json" for d in response.data)


def test_batch_post_bytes(client):
    _check(client.batch_post("/echo", PAYLOADS))


@pytest.mark.anyio
async def test_async_batch_post_bytes(client):
    _check(await client.async_batch_post("/echo", PAYLOADS))


@pytest.mark.parametrize("bad", [b'{"a": ', b"\xff\xfe", b""])
def test_batch_post_rejects_invalid_bytes(client, bad):
    with pytest.raises(ValueError, match="index 1 is not valid UTF-8 JSON"):
        client.batch_post("/echo", [{"ok": 1}, bad])


@pytest.mark.anyio
async def test_async_batch_post_rejects_invalid_bytes(client):
    with pytest.raises(ValueError, match="index 0 is not valid UTF-8 JSON"):
        await client.async_batch_post("/echo", [b"not json"])
