import os
import random
import signal
import socket
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from multiprocessing import Process
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Callable

import psutil
import pytest
import requests

PATCH_PING_MAX_DELAY_SECS = 3


@dataclass
class ControlServerDetails:
    control_server_process: Process
    proxy_server_port: int = 10122
    control_server_port: int = 10123
    inference_server_port: int = 10124

    @property
    def url(self):
        return f"http://localhost:{self.proxy_server_port}"


@pytest.fixture
def control_server(truss_control_container_fs):
    with _configured_control_server(truss_control_container_fs) as server:
        print(truss_control_container_fs)
        yield server


@pytest.mark.integration
def test_truss_control_server_termination(control_server: ControlServerDetails):
    # Port should have been taken up by the servers
    proc_id = control_server.control_server_process.pid
    assert not _is_port_available(control_server.proxy_server_port)
    assert not _is_port_available(control_server.control_server_port)
    assert not _is_port_available(control_server.inference_server_port)

    os.kill(proc_id, signal.SIGTERM)
    control_server.control_server_process.join(timeout=30)
    assert not control_server.control_server_process.is_alive()
    assert _process_tree_is_dead(proc_id)


@pytest.mark.integration
def test_truss_control_server_predict_delays(control_server: ControlServerDetails):
    # Patch to identity code
    identity_model_code = """
class Model:
    def predict(self, model_input):
        return model_input
"""

    _patch(identity_model_code, control_server)

    # run predictions and verify
    num_requests = 100

    def predict(inp):
        time.sleep(random.uniform(0, 0.5))
        resp = requests.post(
            f"{control_server.url}/v1/models/model:predict",
            json=inp,
        )
        return resp.json()

    with ThreadPool(10) as p:
        inputs = list(range(0, num_requests))
        predictions = p.map(predict, inputs)
        assert predictions == inputs


@pytest.mark.integration
def test_truss_control_server_stream(control_server: ControlServerDetails):
    # Patch to identity code
    stream_model_code = """
class Model:
    def predict(self, model_input):
        def inner():
            for i in range(5):
                yield str(i)
        return inner()
"""

    _patch(stream_model_code, control_server)

    resp = requests.post(
        f"{control_server.url}/v1/models/model:predict", json={}, stream=True
    )
    assert resp.headers.get("transfer-encoding") == "chunked"
    assert resp.content == "01234".encode("utf-8")


@pytest.mark.integration
def test_truss_control_server_patch_ping_delays(truss_control_container_fs: Path):
    for _ in range(10):
        with _configured_control_server(
            truss_control_container_fs,
            with_patch_ping_flow=True,
        ) as control_server:
            # Account for patch ping delays
            time.sleep(PATCH_PING_MAX_DELAY_SECS)
            # Port should have been taken up by the servers
            proc_id = control_server.control_server_process.pid
            _assert_with_retry(
                lambda: not _is_port_available(control_server.proxy_server_port),
                "proxy server port is still available",
            )
            _assert_with_retry(
                lambda: not _is_port_available(control_server.control_server_port),
                "control server port is still available",
            )
            _assert_with_retry(
                lambda: not _is_port_available(control_server.inference_server_port),
                "inference server port is still available",
            )

            os.kill(proc_id, signal.SIGTERM)
            control_server.control_server_process.join(timeout=30)
            # Control server process tree should be dead
            assert not control_server.control_server_process.is_alive()
            assert _process_tree_is_dead(proc_id)


def _assert_with_retry(
    pred: Callable[[], bool],
    msg: str,
    retry_interval_secs: float = 0.5,
    retry_count: int = 60,
):
    for _ in range(0, retry_count):
        if pred():
            return
        time.sleep(retry_interval_secs)
    assert False, msg


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


def _process_tree_is_dead(pid: int):
    try:
        proc = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return True

    if proc.is_running():
        return False
    for child_proc in proc.children(recursive=True):
        if child_proc.is_running():
            return False
    return True


@contextmanager
def _configured_control_server(
    truss_control_container_fs: Path,
    with_patch_ping_flow: bool = False,
):
    # TODO(bola): Fix. Now Always set to 8080, since proxy port is not yet configurable
    proxy_port = 8080
    ctrl_port = 8091
    inf_port = 8090
    patch_ping_server_port = 8092

    def start_truss_server(stdout_capture_file_path):
        sys.stdout = open(stdout_capture_file_path, "w")
        app_path = truss_control_container_fs / "app"
        sys.path.append(str(app_path))
        control_path = truss_control_container_fs / "control" / "control"
        sys.path.append(str(control_path))

        env_vars = {
            "CONTROL_SERVER_PORT": f"{ctrl_port}",
            "INFERENCE_SERVER_PORT": f"{inf_port}",
            "APP_HOME": str(app_path),
            "PYTHON_EXECUTABLE_PATH": sys.executable,
        }
        if with_patch_ping_flow:
            env_vars[
                "PATCH_PING_URL_TRUSS"
            ] = f"http://localhost:{patch_ping_server_port}"

        os.environ.update(env_vars)

        # TODO: start hypervisord
        control_start_server_cmd = f'{sys.executable} {str(control_path / "server.py")}'
        nginx_start_cmd = f'nginx -g "daemon off;" -c {str(truss_control_container_fs /"etc/nginx/conf.d/proxy.conf")}'
        supervisord_config_path = (
            truss_control_container_fs / "etc/supervisor/conf.d/supervisord.conf"
        )
        supervisord_config_content = supervisord_config_path.read_text()
        supervisord_config_content = supervisord_config_content.replace(
            """[program:control-server]
command=/control/.env/bin/python3 /control/control/server.py""",
            f"""[program:control-server]
command={control_start_server_cmd}""",
        )
        supervisord_config_content = supervisord_config_content.replace(
            '''[program:nginx]
command=nginx -g "daemon off;"''',
            f"""[program:nginx]
command={nginx_start_cmd}""",
        )
        supervisord_config_path.write_text(supervisord_config_content)
        print(supervisord_config_path.read_text())
        supervidord_cmd = f"supervisord -c {str(supervisord_config_path)}"
        os.system(supervidord_cmd)

    def start_patch_ping_server():
        import json
        import random
        import time
        from http.server import BaseHTTPRequestHandler, HTTPServer

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                time.sleep(random.uniform(0, PATCH_PING_MAX_DELAY_SECS))
                self.send_response(200)
                self.end_headers()
                self.wfile.write(
                    bytes(json.dumps({"is_current": True}), encoding="utf-8")
                )

        httpd = HTTPServer(("localhost", patch_ping_server_port), Handler)
        httpd.serve_forever()

    stdout_capture_file = tempfile.NamedTemporaryFile()
    subproc = Process(target=start_truss_server, args=(stdout_capture_file.name,))
    subproc.start()
    proc_id = subproc.pid
    if with_patch_ping_flow:
        patch_ping_server_proc = Process(target=start_patch_ping_server)
        patch_ping_server_proc.start()
    try:
        time.sleep(2.0)
        # Port should have been taken up by truss server
        yield ControlServerDetails(
            control_server_process=subproc,
            proxy_server_port=proxy_port,
            control_server_port=ctrl_port,
            inference_server_port=inf_port,
        )
    finally:
        if with_patch_ping_flow:
            patch_ping_server_proc.kill()
        # Print on purpose for help with debugging, otherwise hard to know what's going on
        print(Path(stdout_capture_file.name).read_text())
        _kill_process_tree(proc_id)


def _patch(model_code: str, control_server: ControlServerDetails):
    resp = requests.get(f"{control_server.url}/control/truss_hash")
    truss_hash = resp.json()["result"]

    resp = requests.post(
        f"{control_server.url}/control/patch",
        json={
            "hash": "dummy",
            "prev_hash": truss_hash,
            "patches": [
                {
                    "type": "model_code",
                    "body": {
                        "action": "UPDATE",
                        "path": "model.py",
                        "content": model_code,
                    },
                }
            ],
        },
    )
    resp.raise_for_status()
