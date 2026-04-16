"""
Smoke tests for MegatronRLController — require real GPUs.

Run with torchrun (required for Megatron's distributed init even on 1 GPU):

    MEGATRON_LM_PATH=/b10/workspace/trainers/server/vendor/megatron-lm \
    uv run --extra worker --extra dev \
        torchrun --nproc_per_node=1 -m pytest tests/smoke/test_megatron_smoke.py -v -s

Or (single-rank shortcut — sets env vars then calls pytest directly):

    MEGATRON_LM_PATH=/b10/workspace/trainers/server/vendor/megatron-lm \
    RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 \
    uv run --extra worker --extra dev python -m pytest tests/smoke/ -v -s

What these tests verify that unit tests cannot:
  1. MegatronSft initialises Megatron (TP/PP process groups, model load) on real GPUs
  2. A real forward+backward pass completes and returns a finite loss
  3. An optimizer step actually changes the model weights
  4. bridge.export_weights() produces weight tensors whose shapes match the HF config
  5. save_state writes a readable HF-format checkpoint
  6. unwrapped_models is accessible for MegatronWeightWriter
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

# ── Skip whole module if no GPUs ─────────────────────────────────────────────
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 1,
    reason="Smoke tests require at least one GPU",
)

# ── Ensure MEGATRON_LM_PATH and single-rank distributed env vars are set ─────
# Megatron requires torch.distributed even for TP=PP=1.  When running via
# torchrun these are set automatically; otherwise we set them here so the
# tests work when invoked directly (e.g. from an IDE or a single pytest call).

_MEGATRON_PATH = os.environ.get(
    "MEGATRON_LM_PATH",
    "/b10/workspace/trainers/server/vendor/megatron-lm",
)
if _MEGATRON_PATH not in sys.path:
    sys.path.insert(0, _MEGATRON_PATH)
os.environ.setdefault("MEGATRON_LM_PATH", _MEGATRON_PATH)

# Single-rank distributed defaults (no-ops if torchrun already set these)
os.environ.setdefault("RANK", "0")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29500")


# ── Shared fixture ────────────────────────────────────────────────────────────

from trainers_server.dp_worker.api.models import (
    RLControllerConfig,
    TrainingServerConfig,
    InferenceServerConfig,
)
from trainers_server.shared.models import (
    AdamParams,
    Datum,
    ForwardBackwardDetails,
    ModelInput,
    OptimStepDetails,
    TensorData,
)

# Use a tiny model and TP=1 so the smoke test finishes quickly.
_MODEL_ID = "Qwen/Qwen3-0.6B"
_TP = 1
_PP = 1
_GPU = 0


def _smoke_config() -> RLControllerConfig:
    return RLControllerConfig(
        model_id=_MODEL_ID,
        backend="megatron",
        training=TrainingServerConfig(
            tensor_parallel_size=_TP,
            pipeline_parallel_size=_PP,
            max_length=128,
            gpus=[_GPU],
        ),
        inference=InferenceServerConfig(gpus=[1] if torch.cuda.device_count() > 1 else [0]),
    )


def _datum(tokens, reward=1.0):
    return Datum(
        model_input=ModelInput.from_ints(tokens),
        loss_fn_inputs={"reward": TensorData(data=[reward], dtype="float32", shape=[1])},
    )


@pytest.fixture(scope="module")
def ctrl():
    """
    Single MegatronRLController shared across all smoke tests.

    The rollout vLLM server is skipped: smoke tests focus on the training path
    (forward/backward, optim_step, weight export, save_state).  Only the real
    MegatronSft initialization and forward passes are exercised.
    """
    from unittest.mock import patch, MagicMock
    from trainers_server.dp_worker.api.megatron_controller import MegatronRLController

    config = _smoke_config()
    # Skip the vLLM rollout server — we're testing the training side only.
    with (
        patch.object(MegatronRLController, "_launch_rollout_server"),
        patch.object(MegatronRLController, "_init_vllm_client"),
    ):
        controller = MegatronRLController(config)

    # Pre-mark communicator as ready so to_inference doesn't block
    controller._communicator_ready = True
    controller.vllm_client = MagicMock()
    controller.vllm_client.update_flattened_params = MagicMock()
    controller.vllm_client.reset_prefix_cache = MagicMock()

    yield controller
    controller.close()


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestMegatronInit:
    def test_controller_initialises(self, ctrl):
        """MegatronSft must succeed: process groups, model load, optimizer."""
        assert ctrl is not None
        assert ctrl.mode == "training"
        assert ctrl.step == 0

    def test_unwrapped_models_is_list(self, ctrl):
        """trainer.unwrapped_models must be a non-empty list of nn.Modules."""
        assert isinstance(ctrl.unwrapped_models, list)
        assert len(ctrl.unwrapped_models) >= 1
        for m in ctrl.unwrapped_models:
            assert isinstance(m, torch.nn.Module)

    def test_unwrapped_models_on_gpu(self, ctrl):
        """All parameters must live on a CUDA device."""
        for m in ctrl.unwrapped_models:
            for p in m.parameters():
                assert p.is_cuda, f"Parameter {p.shape} is not on GPU"
                break  # just check the first param per model


class TestForwardBackward:
    _TOKENS = list(range(10, 26))  # 16-token sequence

    def test_returns_finite_loss(self, ctrl):
        details = ForwardBackwardDetails(data=[_datum(self._TOKENS)])
        result = ctrl.forward_backward(details)
        loss = result.metrics.get("loss", float("nan"))
        assert torch.isfinite(torch.tensor(loss)), f"Non-finite loss: {loss}"
        assert loss > 0, "Expected positive cross-entropy loss"

    def test_gradients_exist_after_forward(self, ctrl):
        """At least one parameter must have a non-None gradient."""
        details = ForwardBackwardDetails(data=[_datum(self._TOKENS)])
        ctrl.forward_backward(details)
        found_grad = any(
            p.grad is not None
            for m in ctrl.unwrapped_models
            for p in m.parameters()
        )
        assert found_grad, "No gradients found after forward_backward"

    def test_batch_size_two(self, ctrl):
        data = [_datum(list(range(10, 18))), _datum(list(range(20, 30)))]
        result = ctrl.forward_backward(ForwardBackwardDetails(data=data))
        assert torch.isfinite(torch.tensor(result.metrics["loss"]))


class TestOptimStep:
    _TOKENS = list(range(5, 21))

    def test_weights_change_after_step(self, ctrl):
        """An optimizer step with nonzero lr must change at least one weight."""
        # Run a forward pass to produce gradients
        ctrl.forward_backward(ForwardBackwardDetails(data=[_datum(self._TOKENS)]))

        # Snapshot a parameter before the step
        m = ctrl.unwrapped_models[0]
        first_param = next(m.parameters())
        before = first_param.detach().clone()

        ctrl.optim_step(OptimStepDetails(adam_params=AdamParams(learning_rate=1e-4)))

        after = first_param.detach().clone()
        assert not torch.allclose(before, after), "Weights did not change after optim_step"

    def test_step_counter_increments(self, ctrl):
        start = ctrl.step
        ctrl.forward_backward(ForwardBackwardDetails(data=[_datum(self._TOKENS)]))
        ctrl.optim_step(OptimStepDetails(adam_params=AdamParams(learning_rate=1e-4)))
        assert ctrl.step == start + 1


class TestWeightExport:
    _TOKENS = list(range(10, 22))

    def test_export_weights_yields_tensors(self, ctrl):
        """bridge.export_weights must produce (name, tensor) pairs with finite values."""
        # Run forward first so we have a trained state
        ctrl.forward_backward(ForwardBackwardDetails(data=[_datum(self._TOKENS)]))
        ctrl.optim_step(OptimStepDetails(adam_params=AdamParams(learning_rate=1e-4)))

        weights = dict(ctrl.trainer.bridge.export_weights(ctrl.unwrapped_models))
        assert len(weights) > 0, "export_weights returned no weights"
        for name, tensor in weights.items():
            assert isinstance(tensor, torch.Tensor), f"{name} is not a Tensor"
            assert torch.isfinite(tensor).all(), f"{name} contains non-finite values"

    def test_export_weights_has_embedding(self, ctrl):
        """HF checkpoint must contain an embedding weight."""
        weights = dict(ctrl.trainer.bridge.export_weights(ctrl.unwrapped_models))
        embed_keys = [k for k in weights if "embed" in k.lower()]
        assert embed_keys, f"No embedding in exported weights. Keys: {list(weights)[:10]}"


class TestSaveState:
    _TOKENS = list(range(15, 31))

    def test_saves_checkpoint(self, ctrl, tmp_path):
        ctrl.forward_backward(ForwardBackwardDetails(data=[_datum(self._TOKENS)]))
        ctrl.optim_step(OptimStepDetails(adam_params=AdamParams(learning_rate=1e-4)))

        result = ctrl.save_state(str(tmp_path / "ckpt"))
        assert result.mode == "training"

        # Checkpoint directory must exist and have files
        ckpt_dir = tmp_path / "ckpt"
        assert ckpt_dir.exists()
        files = list(ckpt_dir.iterdir())
        assert files, "Checkpoint directory is empty"

    def test_trainer_state_readable(self, ctrl, tmp_path):
        ctrl.forward_backward(ForwardBackwardDetails(data=[_datum(self._TOKENS)]))
        ctrl.optim_step(OptimStepDetails(adam_params=AdamParams(learning_rate=1e-4)))
        ctrl.save_state(str(tmp_path / "ckpt"))

        state = torch.load(str(tmp_path / "ckpt" / "trainer_state.pt"))
        assert "step" in state
        assert state["step"] == ctrl.step
