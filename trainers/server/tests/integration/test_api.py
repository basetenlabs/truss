"""Integration tests for the FastAPI HTTP layer.

Uses ``fastapi.testclient.TestClient`` (synchronous) against the full
``create_app`` / ``RLController`` stack backed by the tiny CPU model from
conftest.  No GPU or ms-swift required.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
from fastapi.testclient import TestClient

from trainers_server.dp_worker.api.server import create_app
from trainers_server.shared.models import (
    AdamParams,
    Datum,
    EncodedTextChunk,
    ForwardBackwardDetails,
    ModelInput,
    OptimStepDetails,
    TensorData,
)


# ---------------------------------------------------------------------------
# Fixture: a TestClient that wraps the real server + tiny model
# ---------------------------------------------------------------------------

@pytest.fixture()
def client(controller):
    app = create_app(controller=controller)
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _datum(tokens, reward=1.0):
    return {
        "model_input": {"chunks": [{"type": "encoded_text", "tokens": tokens}]},
        "loss_fn_inputs": {"reward": {"data": [reward], "dtype": "float32", "shape": [1]}},
    }


def _fb_payload(tokens_list, reward=1.0):
    return {
        "data": [_datum(t, reward) for t in tokens_list],
        "loss_fn": "cross_entropy",
    }


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# /status
# ---------------------------------------------------------------------------

class TestStatus:
    def test_returns_200_with_mode(self, client):
        r = client.get("/status")
        assert r.status_code == 200
        body = r.json()
        assert body["mode"] == "training"
        assert body["step"] == 0

    def test_step_increments_after_optim(self, client):
        client.post("/forward_backward", json=_fb_payload([list(range(10, 20))]))
        client.post("/optim_step", json={"adam_params": {
            "learning_rate": 1e-3, "beta1": 0.9, "beta2": 0.95,
            "eps": 1e-8, "weight_decay": 0.0, "grad_clip_norm": 0.0,
        }})
        r = client.get("/status")
        assert r.json()["step"] == 1


# ---------------------------------------------------------------------------
# /forward_backward
# ---------------------------------------------------------------------------

class TestForwardBackward:
    def test_single_sample(self, client):
        r = client.post("/forward_backward", json=_fb_payload([list(range(10, 20))]))
        assert r.status_code == 200
        body = r.json()
        assert body["loss_fn_output_type"] == "per_token_logprobs"
        assert len(body["loss_fn_outputs"]) == 1
        assert "loss" in body["metrics"]

    def test_multi_sample(self, client):
        tokens_list = [list(range(i, i + 10)) for i in range(3)]
        r = client.post("/forward_backward", json=_fb_payload(tokens_list))
        assert r.status_code == 200
        assert len(r.json()["loss_fn_outputs"]) == 3

    def test_unsupported_loss_fn_returns_400(self, client):
        payload = {"data": [_datum(list(range(10, 20)))], "loss_fn": "mse"}
        r = client.post("/forward_backward", json=payload)
        assert r.status_code == 400
        assert "mse" in r.json()["detail"].lower() or "unsupported" in r.json()["detail"].lower()

    def test_too_short_tokens_returns_400(self, client):
        payload = {"data": [_datum([5])], "loss_fn": "cross_entropy"}
        r = client.post("/forward_backward", json=payload)
        assert r.status_code == 400

    def test_reward_zero_produces_near_zero_loss(self, client):
        r = client.post("/forward_backward", json=_fb_payload([list(range(10, 20))], reward=0.0))
        assert r.status_code == 200
        assert abs(r.json()["metrics"]["loss"]) < 1e-6


# ---------------------------------------------------------------------------
# /optim_step
# ---------------------------------------------------------------------------

class TestOptimStep:
    def _adam_payload(self, lr=1e-3, clip=0.0):
        return {"adam_params": {
            "learning_rate": lr, "beta1": 0.9, "beta2": 0.95,
            "eps": 1e-8, "weight_decay": 0.0, "grad_clip_norm": clip,
        }}

    def test_returns_metrics(self, client):
        client.post("/forward_backward", json=_fb_payload([list(range(10, 20))]))
        r = client.post("/optim_step", json=self._adam_payload())
        assert r.status_code == 200
        body = r.json()
        assert body["metrics"]["step"] == 1
        assert body["metrics"]["learning_rate"] == pytest.approx(1e-3)
        assert body["metrics"]["grad_norm"] > 0

    def test_with_grad_clip(self, client):
        client.post("/forward_backward", json=_fb_payload([list(range(10, 20))]))
        r = client.post("/optim_step", json=self._adam_payload(clip=0.01))
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# /to_training
# ---------------------------------------------------------------------------

class TestToTraining:
    def test_returns_training_mode(self, client, controller):
        controller.mode = "inference"
        r = client.post("/to_training")
        assert r.status_code == 200
        assert r.json()["mode"] == "training"


# ---------------------------------------------------------------------------
# /to_inference
# ---------------------------------------------------------------------------

class TestToInference:
    def test_raises_409_without_communicator(self, client):
        r = client.post("/to_inference")
        assert r.status_code == 409

    def test_succeeds_when_communicator_ready(self, client, controller):
        mock_client = MagicMock()
        mock_client.update_flattened_params = MagicMock()
        mock_client.reset_prefix_cache = MagicMock()
        controller.vllm_client = mock_client
        controller._communicator_ready = True

        r = client.post("/to_inference")
        assert r.status_code == 200
        assert r.json()["mode"] == "inference"


# ---------------------------------------------------------------------------
# /sample
# ---------------------------------------------------------------------------

class TestSample:
    def test_raises_409_when_not_in_inference_mode(self, client):
        payload = {
            "prompt": {"chunks": [{"type": "encoded_text", "tokens": [1, 2, 3]}]},
            "num_samples": 1,
            "sampling_params": {"max_tokens": 10, "temperature": 1.0, "top_p": 1.0},
        }
        r = client.post("/sample", json=payload)
        assert r.status_code == 409


# ---------------------------------------------------------------------------
# /save_state
# ---------------------------------------------------------------------------

class TestSaveState:
    def test_saves_to_path(self, client, tmp_path):
        ckpt = str(tmp_path / "ckpt")
        r = client.post("/save_state", json={"path": ckpt})
        assert r.status_code == 200
        assert (tmp_path / "ckpt" / "trainer_state.pt").exists()

    def test_save_state_preserves_step(self, client, controller, tmp_path):
        client.post("/forward_backward", json=_fb_payload([list(range(10, 20))]))
        client.post("/optim_step", json={"adam_params": {
            "learning_rate": 1e-3, "beta1": 0.9, "beta2": 0.95,
            "eps": 1e-8, "weight_decay": 0.0, "grad_clip_norm": 0.0,
        }})
        ckpt = str(tmp_path / "ckpt")
        client.post("/save_state", json={"path": ckpt})
        state = torch.load(str(tmp_path / "ckpt" / "trainer_state.pt"))
        assert state["step"] == 1


# ---------------------------------------------------------------------------
# Full training loop smoke test
# ---------------------------------------------------------------------------

class TestTrainingLoop:
    def test_forward_backward_optim_cycle(self, client):
        """Simulate 3 steps of an RL training loop."""
        adam = {"adam_params": {
            "learning_rate": 1e-3, "beta1": 0.9, "beta2": 0.95,
            "eps": 1e-8, "weight_decay": 0.0, "grad_clip_norm": 1.0,
        }}
        for i in range(3):
            r = client.post("/forward_backward", json=_fb_payload([list(range(10 + i, 20 + i))]))
            assert r.status_code == 200
            r = client.post("/optim_step", json=adam)
            assert r.status_code == 200
            assert r.json()["metrics"]["step"] == i + 1

        status = client.get("/status").json()
        assert status["step"] == 3
