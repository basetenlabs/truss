import os
import signal
import socket
import sys
import tempfile
import time
from multiprocessing import Process
from pathlib import Path

import yaml


def test_truss_server_termination(truss_container_fs):
    port = 10123

    def start_truss_server(stdout_capture_file_path):
        sys.stdout = open(stdout_capture_file_path, "w")
        app_path = truss_container_fs / "app"
        sys.path.append(str(app_path))

        from common.truss_server import TrussServer

        config = yaml.safe_load((app_path / "config.yaml").read_text())
        server = TrussServer(http_port=port, config=config)
        server.start()

    stdout_capture_file = tempfile.NamedTemporaryFile()
    subproc = Process(target=start_truss_server, args=(stdout_capture_file.name,))
    subproc.start()
    proc_id = subproc.pid
    time.sleep(2.0)
    os.kill(proc_id, signal.SIGTERM)
    time.sleep(2.0)
    print(Path(stdout_capture_file.name).read_text())
    assert not subproc.is_alive()
    assert _is_port_available(port)


def _is_port_available(port):
    try:
        # Try to bind to the given port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("localhost", port))
            return True
    except socket.error:
        # Port is already in use
        return False
