import asyncio
import logging
import multiprocessing
import os
import signal
import socket
import sys
import tempfile
import time
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI
from starlette.responses import PlainTextResponse


def _get_free_port() -> int:
    """Find and return a free port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("localhost", 0))  # Bind to localhost on an arbitrary free port
        return s.getsockname()[1]  # Return the assigned port


HOST = "localhost"
PORT = _get_free_port()


async def _mock_asgi_app():
    await asyncio.sleep(1)
    return PlainTextResponse("OK")


def _on_termination():
    from truss.templates.shared import util

    logging.info("Server is shutting down...")
    util.kill_child_processes(os.getpid())
    os.kill(os.getpid(), signal.SIGKILL)


def run_server(log_file_path: Path):
    import logging

    logging.basicConfig(
        filename=log_file_path,
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Add timestamps
        datefmt="%Y-%m-%d %H:%M:%S",  # Optional: specify date format
        force=True,  # Force reconfiguration of logging if already configured
    )
    import uvicorn
    from truss.templates.server.common.termination_handler_middleware import (
        TerminationHandlerMiddleware,
    )

    app = FastAPI()
    app.get("/")(_mock_asgi_app)
    app.add_middleware(
        TerminationHandlerMiddleware,
        on_termination=_on_termination,
        termination_delay_secs=1,
    )
    # Simple hack to get *all* output to a file.
    sys.stderr = open(log_file_path, "a+")
    sys.stdout = open(log_file_path, "a+")
    uvicorn.run(app, host=HOST, port=PORT)


@pytest.mark.asyncio
async def test_no_outstanding_requests_immediate_termination():
    """Test that the server terminates immediately when no outstanding requests."""
    with tempfile.NamedTemporaryFile(
        delete=False, prefix="test-term.", suffix=".txt"
    ) as tmp_log:
        log_file_path = Path(tmp_log.name)
        server_process = multiprocessing.Process(
            target=run_server, args=(log_file_path,)
        )
        server_process.start()
        time.sleep(1)
        server_process.terminate()
        server_process.join()

        with log_file_path.open() as log:
            log_lines = log.readlines()
            assert any("Received termination signal." in line for line in log_lines)
            assert any(
                "No outstanding requests. Terminate immediately." in line
                for line in log_lines
            )
            assert any("Server is shutting down" in line for line in log_lines)


@pytest.mark.asyncio
async def test_outstanding_requests_delayed_termination():
    """Test that the server waits for outstanding requests to finish before terminating."""
    with tempfile.NamedTemporaryFile(
        delete=False, prefix="test-term.", suffix=".txt"
    ) as tmp_log:
        log_file_path = Path(tmp_log.name)

        server_process = multiprocessing.Process(
            target=run_server, args=(log_file_path,)
        )
        server_process.start()
        time.sleep(1)

        # Send a long-running request to the server
        async with httpx.AsyncClient() as client:
            task = asyncio.create_task(client.get(f"http://{HOST}:{PORT}/"))
            # Give the request some time to be in progress
            await asyncio.sleep(0.5)
            # Send termination signal (SIGTERM) during the request
            server_process.terminate()
            response = await task
            assert response.status_code == 200

        server_process.join()
        with log_file_path.open() as log:
            log_lines = log.readlines()
            assert any("Received termination signal." in line for line in log_lines)
            assert any(
                "Will terminate when all requests are processed." in line
                for line in log_lines
            )
            assert any("Terminating" in line for line in log_lines)
            assert any("Server is shutting down" in line for line in log_lines)


@pytest.mark.asyncio
async def test_multiple_outstanding_requests():
    """Test that the server waits for multiple concurrent requests before terminating.

    Logs something like:

    INFO:     Started server process [1820944]
    INFO:     Waiting for application startup.
    INFO:     Application startup complete.
    INFO:     Uvicorn running on http://localhost:37311 (Press CTRL+C to quit)
    2024-09-30 15:03:05 - root - INFO - Received termination signal.
    2024-09-30 15:03:05 - root - INFO - Will terminate when all requests are processed.
    INFO:     127.0.0.1:58184 - "GET / HTTP/1.1" 200 OK
    INFO:     127.0.0.1:58192 - "GET / HTTP/1.1" 200 OK
    2024-09-30 15:03:06 - root - INFO - Termination after finishing outstanding requests.
    2024-09-30 15:03:06 - root - INFO - Sleeping before termination.
    2024-09-30 15:03:07 - root - INFO - Terminating
    2024-09-30 15:03:07 - root - INFO - Server is shutting down...
    """
    with tempfile.NamedTemporaryFile(
        delete=False, prefix="test-term.", suffix=".txt"
    ) as tmp_log:
        log_file_path = Path(tmp_log.name)

        server_process = multiprocessing.Process(
            target=run_server, args=(log_file_path,)
        )
        server_process.start()
        time.sleep(1)

        # Send multiple concurrent long-running requests
        async with httpx.AsyncClient() as client:
            tasks = [
                asyncio.create_task(client.get(f"http://{HOST}:{PORT}/")),
                asyncio.create_task(client.get(f"http://{HOST}:{PORT}/")),
            ]
            # Give the requests some time to be in progress
            await asyncio.sleep(0.5)
            server_process.terminate()
            # Wait for both requests to finish
            results = await asyncio.gather(*tasks)
            for r in results:
                assert r.status_code == 200

        server_process.join()
        with log_file_path.open() as log:
            log_lines = log.readlines()
            assert any("Received termination signal." in line for line in log_lines)
            assert any(
                "Will terminate when all requests are processed." in line
                for line in log_lines
            )
            assert any("Terminating" in line for line in log_lines)
            assert any("Server is shutting down" in line for line in log_lines)
