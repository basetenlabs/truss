import os
import signal
import socket
import sys
import tempfile
import time
from multiprocessing import Process
from pathlib import Path

import psutil
import pytest
import yaml


@pytest.mark.integration
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
    # Port should have been taken up by truss server
    assert not _is_port_available(port)
    os.kill(proc_id, signal.SIGTERM)
    time.sleep(2.0)
    # Print on purpose for help with debugging, otherwise hard to know what's going on
    print(Path(stdout_capture_file.name).read_text())
    assert not subproc.is_alive()
    # Port should be free now
    assert _is_port_available(port)


@pytest.mark.integration
def test_truss_control_server_termination(truss_control_container_fs):
    ctrl_port = 10123
    inf_port = 10124

    def start_truss_server(stdout_capture_file_path):
        sys.stdout = open(stdout_capture_file_path, "w")
        app_path = truss_control_container_fs / "app"
        sys.path.append(str(app_path))
        control_path = truss_control_container_fs / "control" / "control"
        sys.path.append(str(control_path))

        from server import ControlServer

        control_server = ControlServer(
            python_executable_path=sys.executable,
            inf_serv_home=str(app_path),
            control_server_port=ctrl_port,
            inference_server_port=inf_port,
        )
        control_server.run()

    stdout_capture_file = tempfile.NamedTemporaryFile()
    subproc = Process(target=start_truss_server, args=(stdout_capture_file.name,))
    subproc.start()
    proc_id = subproc.pid
    try:
        time.sleep(2.0)
        # Port should have been taken up by truss server
        assert not _is_port_available(ctrl_port)
        os.kill(proc_id, signal.SIGTERM)
        subproc.join(timeout=10)
        # Print on purpose for help with debugging, otherwise hard to know what's going on
        print(Path(stdout_capture_file.name).read_text())
        assert not subproc.is_alive()
        # Port should be free now
        assert _is_port_available(ctrl_port)
    finally:
        _kill_process_tree(proc_id)


def _is_port_available(port):
    try:
        # Try to bind to the given port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("localhost", port))
            return True
    except socket.error:
        # Port is already in use
        return False


def _kill_process_tree(pid: int):
    try:
        proc = psutil.Process(pid)
        for child_proc in proc.children(recursive=True):
            child_proc.kill()
        proc.kill()
    except psutil.NoSuchProcess:
        pass
