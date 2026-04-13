"""Integration tests for the trainers SDK.

Uses a mock dp_worker (httpx test server) to verify the full training loop:
  1. Forward-backward (gradient accumulation)
  2. Optimizer step
  3. Switch to inference
  4. Sample generations

No external services needed — fully self-contained.
"""

import json
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading

import pytest

from trainers import TrainingClient, AdamParams
from trainers.models import (
    Datum,
    Message,
    ModelInput,
    SampleInput,
    TensorData,
)


class MockWorkerHandler(BaseHTTPRequestHandler):
    """Simulates a dp_worker with mock responses."""

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

        responses = {
            "/forward_backward": {
                "loss_fn_output_type": "scalar",
                "loss_fn_outputs": [],
                "metrics": {"loss": 0.42},
            },
            "/optim_step": {
                "metrics": {"step": 1.0, "learning_rate": 5e-6, "lr": 5e-6},
            },
            "/to_inference": {
                "path": "",
                "mode": "inference",
            },
            "/sample": {
                "sequences": [
                    {"tokens": [1, 2, 3], "text": "four", "logprobs": None},
                ],
                "prompt_logprobs": None,
            },
            "/save_state": {
                "path": "/checkpoints/step-1",
                "mode": "training",
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
    c = TrainingClient(mock_worker, timeout=10.0)
    yield c
    c.close()


# --- Tests ---


def test_health(client):
    client.health()


def test_forward_backward(client):
    batch = [
        Datum(
            model_input=ModelInput.from_ints(list(range(10))),
            loss_fn_inputs={"reward": TensorData.from_list([1.0])},
        ),
    ]
    future = client.forward_backward(batch=batch, loss_fn="cross_entropy")
    result = future.result(timeout=5.0)
    assert result.metrics["loss"] == pytest.approx(0.42)
    assert result.loss_fn_output_type == "scalar"


def test_optim_step(client):
    result = client.optim_step().result(timeout=5.0)
    assert result.metrics["step"] == 1.0
    assert result.metrics["lr"] == pytest.approx(5e-6)


def test_optim_step_with_adam_params(client):
    params = AdamParams(learning_rate=1e-4, beta1=0.9, beta2=0.95)
    result = client.optim_step(params).result(timeout=5.0)
    assert result.metrics is not None


def test_to_inference(client):
    result = client.to_inference().result(timeout=5.0)
    assert result.mode == "inference"


def test_sample(client):
    result = client.sample([
        SampleInput(
            messages=[Message(role="user", content="What is 2+2?")],
            max_tokens=32,
            temperature=0.0,
        ),
    ]).result(timeout=5.0)
    assert len(result.sequences) == 1
    assert result.sequences[0].text == "four"


def test_save_state(client):
    result = client.save_state("/checkpoints/step-1").result(timeout=5.0)
    assert result.path == "/checkpoints/step-1"


def test_pipelining(client):
    """Multiple operations can be dispatched before collecting results."""
    batch = [
        Datum(
            model_input=ModelInput.from_ints(list(range(8))),
            loss_fn_inputs={"reward": TensorData.from_list([1.0])},
        ),
    ]

    # Dispatch 3 forward_backward ops without waiting.
    futures = [
        client.forward_backward(batch=batch)
        for _ in range(3)
    ]

    # Now collect all results.
    for f in futures:
        result = f.result(timeout=5.0)
        assert result.metrics["loss"] == pytest.approx(0.42)


def test_training_loop(client):
    """Full loop: forward_backward x2 → optim_step → to_inference → sample."""
    batch = [
        Datum(
            model_input=ModelInput.from_ints(list(range(10))),
            loss_fn_inputs={"reward": TensorData.from_list([1.0])},
        ),
        Datum(
            model_input=ModelInput.from_ints(list(range(10, 20))),
            loss_fn_inputs={"reward": TensorData.from_list([0.5])},
        ),
    ]

    # Gradient accumulation.
    for _ in range(2):
        result = client.forward_backward(batch=batch).result(timeout=5.0)
        assert "loss" in result.metrics

    # Optimizer step.
    result = client.optim_step(AdamParams(learning_rate=4e-5)).result(timeout=5.0)
    assert "step" in result.metrics

    # Switch to inference.
    result = client.to_inference().result(timeout=5.0)
    assert result.mode == "inference"

    # Sample.
    result = client.sample([
        SampleInput(
            messages=[Message(role="user", content="What is 2+2?")],
            max_tokens=32,
        ),
    ]).result(timeout=5.0)
    assert len(result.sequences) == 1
