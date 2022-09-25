import sys
import threading
import time
from pathlib import Path

import requests


class ServerThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.control_server_host = "0.0.0.0"
        self.control_server_port = 8080

        def configure(flask_app):
            flask_app.config["inference_server_home"] = str(Path(__file__).parent)
            flask_app.config["inference_server_process_args"] = ["sleep", "5"]
            flask_app.config["control_server_host"] = self.control_server_host
            flask_app.config["control_server_port"] = self.control_server_port

        sys.path.append(
            str(
                Path(__file__).parent.parent.parent.parent.parent
                / "templates"
                / "control"
                / "control"
            )
        )
        from truss.templates.control.control.server import _make_server

        self.server = _make_server(configure)

    def run(self):
        self.server.run()

    def shutdown(self):
        print("Shutting down server")
        self.server.close()

    def restart_inference_server(self):
        return self._post("/restart_inference_server")

    def _post(self, path: str = "/"):
        return requests.post(
            f"http://{self.control_server_host}:{self.control_server_port}{path}"
        )


def test_server():
    server_thread = ServerThread()
    server_thread.start()
    try:
        time.sleep(0.1)
        resp = server_thread.restart_inference_server()
        assert resp.status_code == 200
        assert "error" not in resp.json()
        assert "msg" in resp.json()

        # Try second restart
        resp = server_thread.restart_inference_server()
        assert resp.status_code == 200
        assert "error" not in resp.json()
        assert "msg" in resp.json()
    finally:
        server_thread.shutdown()
        print("Waiting for server to shutdown...")
        server_thread.join()
