import gzip
import threading
import zlib
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
import requests

from truss.base.truss_config import ExternalData
from truss.util.download import (
    B10CP_PATH_TRUSS_ENV_VAR_NAME,
    download_external_data,
    download_from_url_using_requests,
)

FILE_CONTENT = b"model data\x00\xff\n" * 10000
GZIP_CONTENT = gzip.compress(FILE_CONTENT, mtime=0)


@pytest.fixture
def download_server():
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/missing":
                self.send_error(404)
                return
            if self.path == "/redirect":
                self.send_response(302)
                self.send_header("Location", "/gzip")
                self.end_headers()
                return

            body, encoding = {
                "/gzip": (GZIP_CONTENT, "gzip"),
                "/deflate": (zlib.compress(FILE_CONTENT), "deflate"),
                "/plain": (FILE_CONTENT, None),
                "/archive": (GZIP_CONTENT, None),
            }[self.path]
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            if encoding is not None:
                self.send_header("Content-Encoding", encoding)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            pass

    with HTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(
            target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        )
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}"
        finally:
            server.shutdown()
            thread.join()


@pytest.mark.parametrize(
    "path,expected",
    [
        ("/gzip", FILE_CONTENT),
        ("/deflate", FILE_CONTENT),
        ("/redirect", FILE_CONTENT),
        ("/plain", FILE_CONTENT),
        ("/archive", GZIP_CONTENT),
    ],
    ids=["gzip-encoding", "deflate-encoding", "redirect", "unencoded", "gzip-file"],
)
def test_external_data_download_content(
    download_server, tmp_path, monkeypatch, path, expected
):
    monkeypatch.delenv(B10CP_PATH_TRUSS_ENV_VAR_NAME, raising=False)
    external_data = ExternalData(
        [{"local_data_path": "nested/weights.bin", "url": download_server + path}]
    )

    download_external_data(external_data, tmp_path)

    assert (tmp_path / "nested" / "weights.bin").read_bytes() == expected


def test_download_http_error_does_not_create_file(download_server, tmp_path):
    destination = tmp_path / "weights.bin"

    with pytest.raises(requests.HTTPError):
        download_from_url_using_requests(download_server + "/missing", destination)

    assert not destination.exists()
