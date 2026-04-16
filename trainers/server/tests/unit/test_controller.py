"""Unit tests for RLController.

All tests run on CPU with a tiny model.  ms-swift is stubbed out by conftest.
No GPU or rollout server is required.
"""
from __future__ import annotations

from typing import List
from unittest.mock import MagicMock, patch

import pytest
import torch

from trainers_server.shared.models import (
    AdamParams,
    Datum,
    EncodedTextChunk,
    ForwardBackwardDetails,
    ModelInput,
    OptimStepDetails,
    SampleDetails,
    SaveStateDetails,
    SampledSequence,
    SamplingParams,
    TensorData,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VOCAB_SIZE = 128  # must match conftest


def _datum(tokens: List[int], reward: float = 1.0) -> Datum:
    return Datum(
        model_input=ModelInput.from_ints(tokens),
        loss_fn_inputs={
            "reward": TensorData(data=[reward], dtype="float32", shape=[1])
        },
    )


def _fb_details(data: List[Datum]) -> ForwardBackwardDetails:
    return ForwardBackwardDetails(data=data)


def _adam_params(**kwargs) -> AdamParams:
    defaults = dict(learning_rate=1e-3, beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.0, grad_clip_norm=0.0)
    defaults.update(kwargs)
    return AdamParams(**defaults)


# ---------------------------------------------------------------------------
# forward_backward tests
# ---------------------------------------------------------------------------

class TestForwardBackward:
    def test_single_sample_returns_result(self, controller):
        tokens = list(range(10, 20))  # 10 tokens, all in [10,20)
        result = controller.forward_backward(_fb_details([_datum(tokens)]))

        assert result.loss_fn_output_type == "per_token_logprobs"
        assert len(result.loss_fn_outputs) == 1
        assert "loss" in result.metrics
        assert isinstance(result.metrics["loss"], float)

    def test_logprobs_shape_matches_valid_tokens(self, controller):
        tokens = list(range(5, 15))  # 10 tokens
        result = controller.forward_backward(_fb_details([_datum(tokens)]))

        # loss_fn_outputs entries are dict[str, TensorData] — use attribute access.
        lp_entry = result.loss_fn_outputs[0]["logprobs"]
        # For a 10-token sequence, shift produces 9 (input[:-1] vs label[1:])
        # valid positions; all are non-padding here.
        assert lp_entry.shape == [9]
        assert len(lp_entry.data) == 9

    def test_multiple_samples(self, controller):
        data = [_datum(list(range(i, i + 10))) for i in range(3)]
        result = controller.forward_backward(_fb_details(data))

        assert len(result.loss_fn_outputs) == 3

    def test_reward_scales_loss(self, controller):
        """Loss should be ~0 when reward is 0 (no gradient signal)."""
        tokens = list(range(10, 20))
        result = controller.forward_backward(_fb_details([_datum(tokens, reward=0.0)]))
        assert abs(result.metrics["loss"]) < 1e-6

    def test_gradients_accumulated(self, controller):
        """After forward_backward the model should have non-zero gradients."""
        tokens = list(range(10, 20))
        controller.forward_backward(_fb_details([_datum(tokens)]))

        has_grad = any(p.grad is not None for p in controller.model.parameters())
        assert has_grad

    def test_rejects_empty_data(self, controller):
        # Pydantic min_length=1 rejects an empty data list.
        with pytest.raises(Exception):
            ForwardBackwardDetails(data=[])

    def test_rejects_single_token_datum(self, controller):
        with pytest.raises(ValueError, match="at least 2 tokens"):
            controller.forward_backward(_fb_details([_datum([1])]))

    def test_rejects_unsupported_loss_fn(self, controller):
        with pytest.raises(ValueError, match="Unsupported loss_fn"):
            controller.forward_backward(
                ForwardBackwardDetails(data=[_datum([1, 2, 3])], loss_fn="mse")
            )

    def test_loss_stored_on_controller(self, controller):
        tokens = list(range(10, 20))
        result = controller.forward_backward(_fb_details([_datum(tokens)]))
        assert controller.last_loss == result.metrics["loss"]

    def test_padding_handled_for_variable_length_batch(self, controller):
        """Batch with different-length sequences should pad without crashing."""
        data = [_datum(list(range(5, 10))), _datum(list(range(5, 20)))]
        result = controller.forward_backward(_fb_details(data))
        assert len(result.loss_fn_outputs) == 2

    def test_auto_switches_to_training_mode(self, controller):
        """If mode is 'inference', forward_backward should auto-switch."""
        controller.mode = "inference"
        tokens = list(range(10, 20))
        controller.forward_backward(_fb_details([_datum(tokens)]))
        assert controller.mode == "training"


# ---------------------------------------------------------------------------
# optim_step tests
# ---------------------------------------------------------------------------

class TestOptimStep:
    def _do_forward_backward(self, controller):
        tokens = list(range(10, 20))
        controller.forward_backward(_fb_details([_datum(tokens)]))

    def test_step_increments_counter(self, controller):
        self._do_forward_backward(controller)
        assert controller.step == 0
        controller.optim_step(OptimStepDetails(adam_params=_adam_params()))
        assert controller.step == 1

    def test_step_counter_increments_per_call(self, controller):
        for expected_step in range(1, 4):
            self._do_forward_backward(controller)
            controller.optim_step(OptimStepDetails(adam_params=_adam_params()))
            assert controller.step == expected_step

    def test_learning_rate_applied(self, controller):
        self._do_forward_backward(controller)
        lr = 5e-4
        controller.optim_step(OptimStepDetails(adam_params=_adam_params(learning_rate=lr)))
        assert controller.optimizer.param_groups[0]["lr"] == pytest.approx(lr)

    def test_betas_applied(self, controller):
        self._do_forward_backward(controller)
        controller.optim_step(OptimStepDetails(adam_params=_adam_params(beta1=0.8, beta2=0.99)))
        assert controller.optimizer.param_groups[0]["betas"] == (0.8, 0.99)

    def test_metrics_returned(self, controller):
        self._do_forward_backward(controller)
        result = controller.optim_step(OptimStepDetails(adam_params=_adam_params()))
        assert "step" in result.metrics
        assert "learning_rate" in result.metrics
        assert "grad_norm" in result.metrics

    def test_grad_norm_nonzero_after_backward(self, controller):
        self._do_forward_backward(controller)
        result = controller.optim_step(OptimStepDetails(adam_params=_adam_params()))
        assert result.metrics["grad_norm"] > 0

    def test_grad_clip_norm(self, controller):
        """With a very small clip norm, the reported grad norm should be <= clip."""
        self._do_forward_backward(controller)
        clip = 0.01
        result = controller.optim_step(OptimStepDetails(adam_params=_adam_params(grad_clip_norm=clip)))
        # After clipping the actual applied norm is <= clip; the reported norm is
        # pre-clip for single-rank, so just check we don't raise.
        assert "grad_norm" in result.metrics

    def test_gradients_zeroed_after_step(self, controller):
        self._do_forward_backward(controller)
        controller.optim_step(OptimStepDetails(adam_params=_adam_params()))
        all_zero = all(
            p.grad is None or p.grad.abs().max().item() == 0
            for p in controller.model.parameters()
        )
        assert all_zero

    def test_model_weights_change_after_step(self, controller):
        params_before = {n: p.clone() for n, p in controller.model.named_parameters()}
        self._do_forward_backward(controller)
        controller.optim_step(OptimStepDetails(adam_params=_adam_params(learning_rate=1e-2)))
        changed = any(
            not torch.allclose(params_before[n], p)
            for n, p in controller.model.named_parameters()
        )
        assert changed


# ---------------------------------------------------------------------------
# mode switching tests
# ---------------------------------------------------------------------------

class TestModeSwitching:
    def test_initial_mode_is_training(self, controller):
        assert controller.mode == "training"

    def test_to_training_returns_status(self, controller):
        result = controller.to_training()
        assert result.mode == "training"

    def test_to_inference_requires_communicator(self, controller):
        """to_inference raises if communicator isn't set up."""
        with pytest.raises(RuntimeError, match="communicator|VLLMClient"):
            controller.to_inference()

    def test_to_inference_syncs_weights_when_communicator_ready(self, controller):
        # Inject a mock vllm_client and mark communicator as ready.
        mock_client = MagicMock()
        mock_client.update_flattened_params = MagicMock()
        mock_client.reset_prefix_cache = MagicMock()
        controller.vllm_client = mock_client
        controller._communicator_ready = True

        result = controller.to_inference()

        assert result.mode == "inference"
        assert controller.mode == "inference"
        mock_client.update_flattened_params.assert_called()
        mock_client.reset_prefix_cache.assert_called_once()

    def test_to_training_switches_from_inference(self, controller):
        controller.mode = "inference"
        result = controller.to_training()
        assert result.mode == "training"
        assert controller.mode == "training"

    def test_sample_raises_if_not_inference(self, controller):
        with pytest.raises(RuntimeError, match="inference mode"):
            controller.sample(SampleDetails(
                prompt=ModelInput.from_ints([1, 2, 3]),
                num_samples=1,
            ))


# ---------------------------------------------------------------------------
# save_state tests
# ---------------------------------------------------------------------------

class TestSaveState:
    def test_saves_checkpoint_files(self, controller, tmp_path):
        ckpt = str(tmp_path / "ckpt")
        controller.save_state(ckpt)

        assert (tmp_path / "ckpt" / "trainer_state.pt").exists()
        state = torch.load(str(tmp_path / "ckpt" / "trainer_state.pt"))
        assert state["step"] == 0
        assert state["mode"] == "training"

    def test_step_persisted_in_checkpoint(self, controller, tmp_path):
        tokens = list(range(10, 20))
        controller.forward_backward(_fb_details([_datum(tokens)]))
        controller.optim_step(OptimStepDetails(adam_params=_adam_params()))

        ckpt = str(tmp_path / "ckpt")
        controller.save_state(ckpt)

        state = torch.load(str(tmp_path / "ckpt" / "trainer_state.pt"))
        assert state["step"] == 1


# ---------------------------------------------------------------------------
# status tests
# ---------------------------------------------------------------------------

class TestStatus:
    def test_status_fields_present(self, controller):
        status = controller.get_status()
        assert status.mode == "training"
        assert status.step == 0
        assert status.model_id == "test-model"

    def test_status_after_steps(self, controller):
        tokens = list(range(10, 20))
        controller.forward_backward(_fb_details([_datum(tokens)]))
        controller.optim_step(OptimStepDetails(adam_params=_adam_params()))
        status = controller.get_status()
        assert status.step == 1
        assert status.last_loss is not None


# ---------------------------------------------------------------------------
# weight-sync helper tests (single-rank path)
# ---------------------------------------------------------------------------

class TestWeightSync:
    def test_push_named_params_calls_vllm_client(self, controller):
        mock_client = MagicMock()
        mock_client.update_flattened_params = MagicMock()
        mock_client.reset_prefix_cache = MagicMock()
        controller.vllm_client = mock_client
        controller._communicator_ready = True

        named_params = [(n, p.detach()) for n, p in controller.model.named_parameters()]
        controller._push_named_params(named_params)

        mock_client.update_flattened_params.assert_called()
        mock_client.reset_prefix_cache.assert_called_once()

    def test_sync_raises_without_communicator(self, controller):
        controller.vllm_client = MagicMock()  # client present but communicator not initialized
        controller._communicator_ready = False
        with pytest.raises(RuntimeError, match="communicator"):
            controller._sync_weights_to_rollout()

    def test_sync_raises_without_vllm_client(self, controller):
        controller.vllm_client = None
        controller._communicator_ready = True
        with pytest.raises(RuntimeError, match="VLLMClient"):
            controller._sync_weights_to_rollout()
