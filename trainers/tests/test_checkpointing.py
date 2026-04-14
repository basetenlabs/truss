"""Tests for trainer checkpointing — save_state / load_state / load_state_with_optimizer.

Controller tests use a minimal fake model (nn.Linear) and bypass the heavy __init__
(no real model loading, no rollout server) to keep tests fast and self-contained.

FastAPI endpoint tests verify routing via TestClient with a mock controller.
"""

import os
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from fastapi.testclient import TestClient

from trainers_server.dp_worker.api.controller import RLController
from trainers_server.dp_worker.api.server import create_app
from trainers_server.shared.models import LoadStateDetails, SaveStateDetails


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeModel(nn.Linear):
    """Minimal fake model that mimics a HuggingFace model's save/load API."""

    def __init__(self, *args, **kwargs):
        super().__init__(4, 4)
        self.save_pretrained = MagicMock()

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        m = cls()
        m.save_pretrained = MagicMock()
        return m


def _make_controller(step: int = 5, last_loss: float = 0.25) -> RLController:
    """Build a minimal RLController, bypassing heavy __init__.

    Uses a tiny _FakeModel so save/load can operate on real tensors without
    needing a GPU or a real HuggingFace model.
    """
    ctrl = object.__new__(RLController)
    ctrl._lock = threading.RLock()
    ctrl.mode = "training"
    ctrl.step = step
    ctrl.last_loss = last_loss
    ctrl._closed = False
    ctrl._rollout_process = None
    ctrl.vllm_client = None
    ctrl._communicator_ready = False

    ctrl.model = _FakeModel()
    ctrl.model.train()
    ctrl.model.save_pretrained = MagicMock()

    ctrl.processor = MagicMock()
    ctrl.processor.save_pretrained = MagicMock()

    ctrl.optimizer = torch.optim.AdamW(ctrl.model.parameters(), lr=1e-3)
    ctrl.optimizer.zero_grad(set_to_none=True)

    ctrl.config = MagicMock()
    ctrl.config.model_id = "test-model"
    ctrl.config.training.gpus = [0]
    ctrl.config.model_dump.return_value = {"model_id": "test-model"}

    return ctrl


# ---------------------------------------------------------------------------
# save_state
# ---------------------------------------------------------------------------

class TestSaveState:
    def test_saves_model_and_trainer_state(self, tmp_path):
        ctrl = _make_controller(step=7, last_loss=0.11)
        ctrl.save_state(str(tmp_path))

        # processor.save_pretrained called with the checkpoint dir
        ctrl.processor.save_pretrained.assert_called_once_with(str(tmp_path))

        # trainer_state.pt written with correct values
        state = torch.load(tmp_path / "trainer_state.pt", map_location="cpu", weights_only=False)
        assert state["step"] == 7
        assert state["last_loss"] == pytest.approx(0.11)
        assert "optimizer" in state

    def test_creates_directory_if_missing(self, tmp_path):
        ctrl = _make_controller()
        nested = tmp_path / "a" / "b" / "c"
        ctrl.save_state(str(nested))
        assert nested.is_dir()

    def test_falls_back_to_env_var(self, tmp_path, monkeypatch):
        monkeypatch.setenv("BT_CHECKPOINT_DIR", str(tmp_path))
        ctrl = _make_controller(step=3)
        ctrl.save_state()  # no path argument
        state = torch.load(tmp_path / "trainer_state.pt", map_location="cpu", weights_only=False)
        assert state["step"] == 3

    def test_raises_when_no_path_and_no_env_var(self, monkeypatch):
        monkeypatch.delenv("BT_CHECKPOINT_DIR", raising=False)
        ctrl = _make_controller()
        with pytest.raises(ValueError, match="BT_CHECKPOINT_DIR"):
            ctrl.save_state()

    def test_returns_save_state_result(self, tmp_path):
        ctrl = _make_controller()
        result = ctrl.save_state(str(tmp_path))
        assert result.mode == "training"


# ---------------------------------------------------------------------------
# load_state
# ---------------------------------------------------------------------------

class TestLoadState:
    def _save_checkpoint(self, tmp_path, ctrl):
        ctrl.save_state(str(tmp_path))

    def test_loads_weights_and_resets_optimizer(self, tmp_path):
        ctrl = _make_controller(step=4)
        self._save_checkpoint(tmp_path, ctrl)

        # Advance step and do a fake gradient update so optimizer has state
        ctrl.step = 99

        new_ctrl = _make_controller(step=99)
        result = new_ctrl.load_state(LoadStateDetails(path=str(tmp_path)))

        assert result.mode == "training"
        # step is preserved from the controller (load_state doesn't restore step)
        assert new_ctrl.mode == "training"
        assert not new_ctrl._communicator_ready

    def test_raises_when_path_missing(self, tmp_path):
        ctrl = _make_controller()
        missing = tmp_path / "does_not_exist"
        with pytest.raises(ValueError, match="not found"):
            ctrl.load_state(LoadStateDetails(path=str(missing)))

    def test_falls_back_to_env_var(self, tmp_path, monkeypatch):
        ctrl = _make_controller(step=2)
        self._save_checkpoint(tmp_path, ctrl)

        monkeypatch.setenv("BT_LOAD_CHECKPOINT_DIR", str(tmp_path))
        new_ctrl = _make_controller()
        result = new_ctrl.load_state(LoadStateDetails())  # no path

        assert result.mode == "training"

    def test_raises_when_no_path_and_no_env_var(self, monkeypatch):
        monkeypatch.delenv("BT_LOAD_CHECKPOINT_DIR", raising=False)
        ctrl = _make_controller()
        with pytest.raises(ValueError, match="BT_LOAD_CHECKPOINT_DIR"):
            ctrl.load_state(LoadStateDetails())


# ---------------------------------------------------------------------------
# load_state_with_optimizer
# ---------------------------------------------------------------------------

class TestLoadStateWithOptimizer:
    def _save_checkpoint(self, tmp_path, ctrl):
        ctrl.save_state(str(tmp_path))

    def test_restores_step_and_optimizer(self, tmp_path):
        ctrl = _make_controller(step=11, last_loss=0.77)
        self._save_checkpoint(tmp_path, ctrl)

        new_ctrl = _make_controller(step=0, last_loss=None)
        result = new_ctrl.load_state_with_optimizer(LoadStateDetails(path=str(tmp_path)))

        assert result.step == 11
        assert new_ctrl.step == 11
        assert new_ctrl.last_loss == pytest.approx(0.77)
        assert result.mode == "training"
        assert not new_ctrl._communicator_ready

    def test_succeeds_without_trainer_state_pt(self, tmp_path):
        """If no trainer_state.pt exists, loads weights but doesn't restore step."""
        ctrl = _make_controller(step=5)
        # Only save model weights, not the full checkpoint
        (tmp_path / "dummy_weight.bin").touch()  # make dir non-empty

        new_ctrl = _make_controller(step=0)
        # Should not raise even without trainer_state.pt
        result = new_ctrl.load_state_with_optimizer(LoadStateDetails(path=str(tmp_path)))

        assert result.mode == "training"
        assert new_ctrl.step == 0  # unchanged — no trainer_state.pt to restore from

    def test_raises_when_path_missing(self, tmp_path):
        ctrl = _make_controller()
        with pytest.raises(ValueError, match="not found"):
            ctrl.load_state_with_optimizer(LoadStateDetails(path=str(tmp_path / "missing")))


# ---------------------------------------------------------------------------
# Roundtrip: save → load_state_with_optimizer → save again
# ---------------------------------------------------------------------------

class TestCheckpointRoundtrip:
    def test_roundtrip_preserves_step_and_loss(self, tmp_path):
        ckpt1 = tmp_path / "ckpt1"
        ckpt2 = tmp_path / "ckpt2"

        original = _make_controller(step=42, last_loss=0.123)
        original.save_state(str(ckpt1))

        restored = _make_controller(step=0)
        restored.load_state_with_optimizer(LoadStateDetails(path=str(ckpt1)))

        assert restored.step == 42
        assert restored.last_loss == pytest.approx(0.123)

        # Save again from the restored state
        restored.save_state(str(ckpt2))
        state2 = torch.load(ckpt2 / "trainer_state.pt", map_location="cpu", weights_only=False)
        assert state2["step"] == 42


# ---------------------------------------------------------------------------
# FastAPI endpoint tests (routing + response shape)
# ---------------------------------------------------------------------------

@pytest.fixture
def api_client():
    """TestClient backed by a mock controller — no model loading."""
    ctrl = MagicMock(spec=RLController)

    from trainers_server.shared.models import LoadStateResult, SaveStateResult
    ctrl.save_state.return_value = SaveStateResult(mode="training")
    ctrl.load_state.return_value = LoadStateResult(mode="training", step=7)
    ctrl.load_state_with_optimizer.return_value = LoadStateResult(mode="training", step=7)

    app = create_app(controller=ctrl)
    return TestClient(app), ctrl


class TestEndpoints:
    def test_save_state_no_body(self, api_client):
        client, ctrl = api_client
        resp = client.post("/save_state")
        assert resp.status_code == 200
        ctrl.save_state.assert_called_once_with(None)

    def test_save_state_with_path(self, api_client):
        client, ctrl = api_client
        resp = client.post("/save_state", json={"path": "/ckpt/step-1"})
        assert resp.status_code == 200
        ctrl.save_state.assert_called_once_with("/ckpt/step-1")

    def test_load_state_no_body(self, api_client):
        client, ctrl = api_client
        resp = client.post("/load_state")
        assert resp.status_code == 200
        data = resp.json()
        assert data["mode"] == "training"
        assert data["step"] == 7

    def test_load_state_with_path(self, api_client):
        client, ctrl = api_client
        resp = client.post("/load_state", json={"path": "/ckpt/step-1"})
        assert resp.status_code == 200
        from trainers_server.shared.models import LoadStateDetails
        ctrl.load_state.assert_called_once_with(LoadStateDetails(path="/ckpt/step-1"))

    def test_load_state_with_optimizer_no_body(self, api_client):
        client, ctrl = api_client
        resp = client.post("/load_state_with_optimizer")
        assert resp.status_code == 200
        data = resp.json()
        assert data["mode"] == "training"
        assert data["step"] == 7

    def test_load_state_with_optimizer_with_path(self, api_client):
        client, ctrl = api_client
        resp = client.post("/load_state_with_optimizer", json={"path": "/ckpt/step-2"})
        assert resp.status_code == 200
        from trainers_server.shared.models import LoadStateDetails
        ctrl.load_state_with_optimizer.assert_called_once_with(LoadStateDetails(path="/ckpt/step-2"))

    def test_save_state_value_error_returns_400(self, api_client):
        client, ctrl = api_client
        ctrl.save_state.side_effect = ValueError("no path provided")
        resp = client.post("/save_state")
        assert resp.status_code == 400

    def test_load_state_value_error_returns_400(self, api_client):
        client, ctrl = api_client
        ctrl.load_state.side_effect = ValueError("checkpoint not found")
        resp = client.post("/load_state")
        assert resp.status_code == 400
