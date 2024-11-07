import os
import signal
import socket
import sys
import tempfile
import time
from multiprocessing import Process
from pathlib import Path

import pytest


@pytest.mark.integration
def test_truss_server_termination(truss_container_fs):
    port = 10123

    def start_truss_server(stdout_capture_file_path):
        sys.stdout = open(stdout_capture_file_path, "w")
        app_path = truss_container_fs / "app"
        sys.path.append(str(app_path))
        os.chdir(app_path)

        from truss_server import TrussServer

        server = TrussServer(http_port=port, config_or_path=app_path / "config.yaml")
        server.start()

    stdout_capture_file = tempfile.NamedTemporaryFile()
    subproc = Process(target=start_truss_server, args=(stdout_capture_file.name,))
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


def _is_port_available(port):
    try:
        # Try to bind to the given port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("localhost", port))
            return True
    except socket.error:
        # Port is already in use
        return False
