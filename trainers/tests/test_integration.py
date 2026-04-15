"""Integration tests for the trainers SDK.

Uses a mock dp_worker (httpx test server) to verify the full training loop:
  1. Forward-backward (gradient accumulation)
  2. Optimizer step
  3. Switch to inference
  4. Sample generations

No external services needed — fully self-contained.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from trainers import (
    AdamParams,
    Datum,
    ModelInput,
    SamplingParams,
    ServiceClient,
    TensorData,
)


class MockWorkerHandler(BaseHTTPRequestHandler):
    """Simulates a dp_worker with mock responses."""

    last_request: dict = {}  # class-level, stores most recent POST body per path

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.end_headers()
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length else b""
        if body:
            MockWorkerHandler.last_request[self.path] = json.loads(body)

        responses = {
            "/forward_backward": {
                "loss_fn_output_type": "scalar",
                "loss_fn_outputs": [],
                "metrics": {"loss": 0.42},
            },
            "/optim_step": {
                "metrics": {"step": 1.0, "learning_rate": 5e-6, "lr": 5e-6}
            },
            "/to_inference": {"path": "", "type": "save_weights"},
            "/sample": {
                "sequences": [
                    {
                        "tokens": [1, 2, 3],
                        "stop_reason": "stop",
                        "logprobs": [-0.1, -0.2, -0.3],
                    }
                ]
            },
            "/save_state": {"path": "step-1", "type": "save_weights"},
            "/load_state": {"path": "/ckpt/step-1", "type": "load_weights"},
            "/load_state_with_optimizer": {
                "path": "/ckpt/step-1",
                "type": "load_weights",
            },
        }

        resp_body = responses.get(self.path)
        if resp_body is not None:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(resp_body).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # suppress logs


@pytest.fixture(scope="module")
def mock_worker():
    server = HTTPServer(("127.0.0.1", 0), MockWorkerHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()


@pytest.fixture
def client(mock_worker):
    service = ServiceClient(base_url=mock_worker)
    c = service.create_lora_training_client(base_model="test-model", timeout=10.0)
    yield c
    c.close()


# --- Tests ---


def test_health(client):
    client.health()


def test_forward_backward(client):
    batch = [
        Datum(
            model_input=ModelInput.from_ints(list(range(10))),
            loss_fn_inputs={
                "reward": TensorData(data=[1.0], dtype="float32", shape=[1])
            },
        )
    ]
    future = client.forward_backward(data=batch, loss_fn="cross_entropy")
    result = future.result(timeout=5.0)
    assert result.metrics["loss"] == pytest.approx(0.42)
    assert result.loss_fn_output_type == "scalar"


def test_optim_step(client):
    result = client.optim_step(AdamParams()).result(timeout=5.0)
    assert result.metrics["step"] == 1.0
    assert result.metrics["lr"] == pytest.approx(5e-6)


def test_optim_step_with_adam_params(client):
    params = AdamParams(learning_rate=1e-4, beta1=0.9, beta2=0.95)
    result = client.optim_step(params).result(timeout=5.0)
    assert result.metrics is not None


def test_save_weights_and_get_sampling_client(client):
    result = client.save_weights_and_get_sampling_client().result(timeout=5.0)
    assert result.type == "save_weights"


def test_sample(client):
    result = client.sample(
        prompt=ModelInput.from_ints([1, 2, 3]),
        num_samples=1,
        sampling_params=SamplingParams(max_tokens=32, temperature=0.0),
    ).result(timeout=5.0)
    assert len(result.sequences) == 1
    assert result.sequences[0].tokens == [1, 2, 3]
    assert result.sequences[0].logprobs == [-0.1, -0.2, -0.3]
    assert result.sequences[0].stop_reason == "stop"


def test_save_state(client):
    result = client.save_state("step-1").result(timeout=5.0)
    assert result.path == "step-1"


def test_pipelining(client):
    """Multiple operations can be dispatched before collecting results."""
    batch = [
        Datum(
            model_input=ModelInput.from_ints(list(range(8))),
            loss_fn_inputs={
                "reward": TensorData(data=[1.0], dtype="float32", shape=[1])
            },
        )
    ]

    # Dispatch 3 forward_backward ops without waiting.
    futures = [client.forward_backward(data=batch) for _ in range(3)]

    # Now collect all results.
    for f in futures:
        result = f.result(timeout=5.0)
        assert result.metrics["loss"] == pytest.approx(0.42)


def test_training_loop(client):
    """Full loop: forward_backward x2 → optim_step → to_inference → sample."""
    batch = [
        Datum(
            model_input=ModelInput.from_ints(list(range(10))),
            loss_fn_inputs={
                "reward": TensorData(data=[1.0], dtype="float32", shape=[1])
            },
        ),
        Datum(
            model_input=ModelInput.from_ints(list(range(10, 20))),
            loss_fn_inputs={
                "reward": TensorData(data=[0.5], dtype="float32", shape=[1])
            },
        ),
    ]

    # Gradient accumulation.
    for _ in range(2):
        result = client.forward_backward(data=batch).result(timeout=5.0)
        assert "loss" in result.metrics

    # Optimizer step.
    result = client.optim_step(AdamParams(learning_rate=4e-5)).result(timeout=5.0)
    assert "step" in result.metrics

    # Switch to inference.
    result = client.save_weights_and_get_sampling_client().result(timeout=5.0)
    assert result.type == "save_weights"

    # Sample.
    result = client.sample(
        prompt=ModelInput.from_ints([1, 2, 3]),
        sampling_params=SamplingParams(max_tokens=32),
    ).result(timeout=5.0)
    assert len(result.sequences) == 1


# --- load_state / load_state_with_optimizer ---


def test_load_state(client):
    result = client.load_state("/ckpt/step-1").result(timeout=5.0)
    assert result.type == "load_weights"
    assert MockWorkerHandler.last_request["/load_state"]["path"] == "/ckpt/step-1"


def test_load_state_with_optimizer(client):
    result = client.load_state_with_optimizer("/ckpt/step-1").result(timeout=5.0)
    assert result.type == "load_weights"
    assert (
        MockWorkerHandler.last_request["/load_state_with_optimizer"]["path"]
        == "/ckpt/step-1"
    )


# --- ServiceClient factory methods ---


def test_create_training_client_from_state(mock_worker):
    service = ServiceClient(base_url=mock_worker)
    c = service.create_training_client_from_state("/ckpt/step-1")
    try:
        c.health()  # verifies the returned client points at a live server
    finally:
        c.close()


def test_create_training_client_from_state_with_optimizer(mock_worker):
    service = ServiceClient(base_url=mock_worker)
    c = service.create_training_client_from_state_with_optimizer("/ckpt/step-1")
    try:
        c.health()
    finally:
        c.close()


def test_create_training_client_from_state_accepts_explicit_base_url(mock_worker):
    service = ServiceClient(base_url="http://unused:9999")
    c = service.create_training_client_from_state("/ckpt/step-1", base_url=mock_worker)
    try:
        c.health()  # would fail if it used the unused URL
    finally:
        c.close()
