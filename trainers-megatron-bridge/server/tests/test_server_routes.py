"""Tests for the FastAPI HTTP layer (server.py).

Uses FastAPI's TestClient with a mocked RLController. No GPU required.
Verifies that each route:
  - Deserializes the request body into the right Pydantic type
  - Calls the correct controller method
  - Returns the correct status code and response shape
  - Converts ValueError → 400 and RuntimeError → 409
"""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from trainers_server.dp_worker.api.controller import RLController
from trainers_server.dp_worker.api.models import RLControllerConfig, StatusResult
from trainers_server.dp_worker.api.server import create_app
from trainers_server.shared.models import (
    ForwardBackwardResult,
    OptimStepResult,
    SampleResult,
    SampledSequence,
    SaveStateResult,
    ToInferenceResult,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_ctrl():
    ctrl = MagicMock(spec=RLController)

    ctrl.forward_backward.return_value = ForwardBackwardResult(
        loss_fn_output_type="per_token_logprobs",
        loss_fn_outputs=[],
        metrics={"loss": 0.42},
    )
    ctrl.optim_step.return_value = OptimStepResult(
        metrics={"step": 1.0, "learning_rate": 1e-4, "lr": 1e-4, "grad_norm": 0.123}
    )
    ctrl.to_inference.return_value = ToInferenceResult(mode="inference")
    ctrl.to_training.return_value = StatusResult(
        mode="training", step=0, model_id="test", device="cpu"
    )
    ctrl.sample.return_value = SampleResult(
        sequences=[SampledSequence(tokens=[10, 20, 30], stop_reason="stop")]
    )
    ctrl.save_state.return_value = SaveStateResult(mode="training")
    ctrl.get_status.return_value = StatusResult(
        mode="training", step=2, model_id="test", device="cuda:0", last_loss=0.5,
        gpu_memory={"cuda:0": 1024}
    )
    return ctrl


@pytest.fixture
def client(mock_ctrl):
    app = create_app(controller=mock_ctrl)
    return TestClient(app)


# ── Health ────────────────────────────────────────────────────────────────────


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200


# ── Status ────────────────────────────────────────────────────────────────────


def test_get_status(client, mock_ctrl):
    r = client.get("/status")
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "training"
    assert body["step"] == 2
    assert body["last_loss"] == pytest.approx(0.5)
    mock_ctrl.get_status.assert_called_once()


# ── /forward_backward ─────────────────────────────────────────────────────────


def test_forward_backward_returns_loss(client, mock_ctrl):
    payload = {
        "data": [
            {
                "model_input": {"chunks": [{"type": "encoded_text", "tokens": [1, 2, 3, 4]}]},
                "loss_fn_inputs": {"reward": {"data": [1.0], "dtype": "float32", "shape": [1]}},
            }
        ],
        "loss_fn": "cross_entropy",
    }
    r = client.post("/forward_backward", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["metrics"]["loss"] == pytest.approx(0.42)
    assert body["loss_fn_output_type"] == "per_token_logprobs"
    mock_ctrl.forward_backward.assert_called_once()


def test_forward_backward_bad_loss_fn_returns_400(client, mock_ctrl):
    mock_ctrl.forward_backward.side_effect = ValueError("Unsupported loss_fn")
    payload = {
        "data": [
            {
                "model_input": {"chunks": [{"type": "encoded_text", "tokens": [1, 2]}]},
                "loss_fn_inputs": {},
            }
        ],
        "loss_fn": "unsupported_fn",
    }
    r = client.post("/forward_backward", json=payload)
    assert r.status_code == 400


def test_forward_backward_runtime_error_returns_409(client, mock_ctrl):
    mock_ctrl.forward_backward.side_effect = RuntimeError("not in training mode")
    payload = {
        "data": [
            {
                "model_input": {"chunks": [{"type": "encoded_text", "tokens": [1, 2]}]},
                "loss_fn_inputs": {},
            }
        ],
    }
    r = client.post("/forward_backward", json=payload)
    assert r.status_code == 409


# ── /optim_step ───────────────────────────────────────────────────────────────


def test_optim_step_with_explicit_params(client, mock_ctrl):
    payload = {"adam_params": {"learning_rate": 1e-4, "beta1": 0.9, "beta2": 0.95}}
    r = client.post("/optim_step", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["metrics"]["step"] == pytest.approx(1.0)
    assert body["metrics"]["grad_norm"] == pytest.approx(0.123)
    mock_ctrl.optim_step.assert_called_once()


def test_optim_step_minimal_body(client, mock_ctrl):
    """Empty adam_params dict should use all defaults."""
    r = client.post("/optim_step", json={"adam_params": {}})
    assert r.status_code == 200


# ── /to_inference ─────────────────────────────────────────────────────────────


def test_to_inference(client, mock_ctrl):
    r = client.post("/to_inference")
    assert r.status_code == 200
    assert r.json()["mode"] == "inference"
    mock_ctrl.to_inference.assert_called_once()


# ── /to_training ──────────────────────────────────────────────────────────────


def test_to_training(client, mock_ctrl):
    r = client.post("/to_training")
    assert r.status_code == 200
    assert r.json()["mode"] == "training"
    mock_ctrl.to_training.assert_called_once()


# ── /sample ───────────────────────────────────────────────────────────────────


def test_sample(client, mock_ctrl):
    payload = {
        "prompt": {"chunks": [{"type": "encoded_text", "tokens": [1, 2, 3]}]},
        "num_samples": 2,
        "sampling_params": {"max_tokens": 64, "temperature": 0.7},
    }
    r = client.post("/sample", json=payload)
    assert r.status_code == 200
    seqs = r.json()["sequences"]
    assert len(seqs) == 1
    assert seqs[0]["tokens"] == [10, 20, 30]
    assert seqs[0]["stop_reason"] == "stop"
    mock_ctrl.sample.assert_called_once()


def test_sample_runtime_error_returns_409(client, mock_ctrl):
    mock_ctrl.sample.side_effect = RuntimeError("sample() is only valid in inference mode.")
    payload = {
        "prompt": {"chunks": [{"type": "encoded_text", "tokens": [1]}]},
    }
    r = client.post("/sample", json=payload)
    assert r.status_code == 409


# ── /save_state ───────────────────────────────────────────────────────────────


def test_save_state(client, mock_ctrl):
    payload = {"path": "/tmp/checkpoint-step-1"}
    r = client.post("/save_state", json=payload)
    assert r.status_code == 200
    assert r.json()["mode"] == "training"
    # Verify the controller was called with the path string
    mock_ctrl.save_state.assert_called_once_with("/tmp/checkpoint-step-1")


# ── Full mini loop ────────────────────────────────────────────────────────────


def test_full_mini_loop(client, mock_ctrl):
    """forward_backward → optim_step → to_inference → sample."""
    datum_payload = {
        "model_input": {"chunks": [{"type": "encoded_text", "tokens": list(range(10))}]},
        "loss_fn_inputs": {"reward": {"data": [1.0], "dtype": "float32", "shape": [1]}},
    }

    # Forward-backward x2 (gradient accumulation).
    for _ in range(2):
        r = client.post("/forward_backward", json={"data": [datum_payload]})
        assert r.status_code == 200

    # Optimizer step.
    r = client.post("/optim_step", json={"adam_params": {"learning_rate": 5e-5}})
    assert r.status_code == 200
    assert r.json()["metrics"]["step"] == pytest.approx(1.0)

    # Switch to inference.
    r = client.post("/to_inference")
    assert r.status_code == 200
    assert r.json()["mode"] == "inference"

    # Sample.
    r = client.post("/sample", json={
        "prompt": {"chunks": [{"type": "encoded_text", "tokens": [1, 2, 3]}]},
        "num_samples": 1,
    })
    assert r.status_code == 200
    assert len(r.json()["sequences"]) == 1
