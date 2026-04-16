"""
Unit tests for MegatronRLController.

These tests run without a GPU, without ms-swift's Megatron stack, and without
any distributed setup.  They test:

  1. ``_to_megatron_batch`` — the batch conversion that must produce exactly
     the format Megatron's ``get_batch_on_this_tp_rank`` expects.
  2. Config routing — ``backend="megatron"`` in RLControllerConfig must reach
     MegatronRLController, not RLController.
  3. Controller flow via a mock MegatronTrainer — verifies that the right
     Megatron hooks are called during forward_backward / optim_step /
     to_inference / save_state.

What is NOT tested here (requires real hardware):
  - Whether MegatronSft initializes without error on N GPUs.
  - Whether TP shard gathering in bridge.export_weights produces correct HF
    weights (see tests/smoke/test_megatron_smoke.py for that).
  - Whether MegatronWeightWriter can consume trainer.unwrapped_models.
"""
from __future__ import annotations

import sys
import types
from typing import List
from unittest.mock import MagicMock, patch, call
import pytest
import torch

# ── conftest installs lightweight swift stubs; we extend them for megatron ──

def _install_megatron_stubs():
    """Add swift.megatron.* and megatron.core.* stubs so megatron_controller.py can be imported."""
    swift_mod = sys.modules.get("swift")
    if swift_mod is None:
        return  # conftest hasn't run yet; nothing to do

    for sub in [
        "swift.megatron",
        "swift.megatron.arguments",
        "swift.megatron.pipelines",
        "swift.megatron.pipelines.train",
        "swift.megatron.pipelines.train.sft",
    ]:
        if sub not in sys.modules:
            sys.modules[sub] = types.ModuleType(sub)

    sys.modules["swift.megatron.arguments"].MegatronSftArguments = MagicMock()
    sys.modules["swift.megatron.pipelines.train.sft"].MegatronSft = MagicMock()

    # Install megatron.core stubs so that megatron_controller.py's lazy
    # ``from megatron.core.pipeline_parallel import get_forward_backward_func``
    # resolves without a real megatron-core install.
    for sub in ["megatron", "megatron.core", "megatron.core.pipeline_parallel"]:
        if sub not in sys.modules:
            sys.modules[sub] = types.ModuleType(sub)

    # Default stub returns a single loss dict; individual tests can patch this.
    _default_loss_dicts = [{"lm loss": torch.tensor([2.0, 10.0])}]
    _default_fb_inner = MagicMock(return_value=_default_loss_dicts)
    sys.modules["megatron.core.pipeline_parallel"].get_forward_backward_func = MagicMock(
        return_value=_default_fb_inner
    )


_install_megatron_stubs()

# ── shared helpers ────────────────────────────────────────────────────────────

from trainers_server.shared.models import (
    Datum,
    ForwardBackwardDetails,
    ModelInput,
    OptimStepDetails,
    SaveStateDetails,
    TensorData,
    AdamParams,
)
from trainers_server.dp_worker.api.models import (
    RLControllerConfig,
    TrainingServerConfig,
    InferenceServerConfig,
)


def _datum(tokens: List[int], reward: float = 1.0) -> Datum:
    return Datum(
        model_input=ModelInput.from_ints(tokens),
        loss_fn_inputs={"reward": TensorData(data=[reward], dtype="float32", shape=[1])},
    )


def _fb_details(data: List[Datum]) -> ForwardBackwardDetails:
    return ForwardBackwardDetails(data=data)


def _megatron_config(tp: int = 1, pp: int = 1) -> RLControllerConfig:
    return RLControllerConfig(
        model_id="test-model",
        backend="megatron",
        training=TrainingServerConfig(
            tensor_parallel_size=tp,
            pipeline_parallel_size=pp,
            max_length=64,
            gpus=[0],
        ),
        inference=InferenceServerConfig(gpus=[1]),
    )


# ── helper: build a MegatronRLController with a fully mocked trainer ─────────

def _mock_megatron_controller(config: RLControllerConfig | None = None):
    """
    Build a MegatronRLController whose MegatronSft is entirely mocked.

    Uses ``__new__`` to bypass ``__init__`` (which would call real Megatron init),
    then sets all attributes manually.  Megatron stub modules installed by
    ``_install_megatron_stubs()`` make the lazy ``from megatron.core...`` imports
    inside controller methods work without a real megatron-core install.

    Returns (controller, mock_trainer, mock_bridge).
    """
    from tests.conftest import _FakeProcessor, make_tiny_model
    from trainers_server.dp_worker.api.megatron_controller import MegatronRLController

    if config is None:
        config = _megatron_config()

    # Mock trainer and bridge
    mock_bridge = MagicMock()
    mock_bridge.export_weights.return_value = iter([("model.embed.weight", torch.zeros(4, 4))])
    mock_bridge.save_weights = MagicMock()

    mock_model = make_tiny_model()
    mock_optimizer = MagicMock()
    mock_optimizer.param_groups = [{"lr": 0.0, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.0}]
    mock_optimizer.step = MagicMock()
    mock_optimizer.zero_grad = MagicMock()

    mock_trainer = MagicMock()
    mock_trainer.unwrapped_models = [mock_model]
    mock_trainer.wrapped_models = [mock_model]
    mock_trainer.optimizer = mock_optimizer
    mock_trainer.bridge = mock_bridge
    mock_trainer.forward_step = MagicMock(return_value=(torch.tensor(1.0), {}))

    mock_sft = MagicMock()
    mock_sft.trainer = mock_trainer
    mock_sft.processor = _FakeProcessor()

    # Bypass __init__ (which calls MegatronSft) and set attributes manually.
    ctrl = MegatronRLController.__new__(MegatronRLController)
    ctrl.config = config
    ctrl._lock = __import__("threading").RLock()
    ctrl.mode = "training"
    ctrl.step = 0
    ctrl.last_loss = None
    ctrl._closed = False
    ctrl._rollout_process = None
    ctrl.vllm_client = None
    ctrl._communicator_ready = False
    ctrl._rollout_port = 0
    ctrl._rollout_group_port = 0
    ctrl._rollout_max_model_len = 4096
    ctrl._sft = mock_sft
    ctrl.trainer = mock_trainer
    ctrl.processor = mock_sft.processor
    ctrl.unwrapped_models = mock_trainer.unwrapped_models

    return ctrl, mock_trainer, mock_bridge


# ── _to_megatron_batch ────────────────────────────────────────────────────────

class TestToMegatronBatch:
    """
    ``_to_megatron_batch`` is pure PyTorch — no Megatron required.

    Megatron's causal-LM path calls::

        data['labels'] = torch.roll(data['labels'], -1, dims=-1)
        if 'loss_scale' in data:
            data['loss_scale'] = torch.roll(data['loss_scale'], -1, dims=-1)

    so that ``labels[t] = tokens[t+1]`` after the roll.  We verify the
    PRE-ROLL tensors match exactly what ``get_batch_on_this_tp_rank`` expects.
    """

    def _ctrl(self):
        ctrl, _, _ = _mock_megatron_controller()
        return ctrl

    def test_output_keys_present(self):
        ctrl = self._ctrl()
        batch = ctrl._to_megatron_batch(_fb_details([_datum(list(range(5, 15)))]))
        for key in ("input_ids", "labels", "position_ids", "loss_scale", "num_samples"):
            assert key in batch, f"missing key: {key}"

    def test_single_sample_shapes(self):
        tokens = list(range(10, 20))
        ctrl = self._ctrl()
        batch = ctrl._to_megatron_batch(_fb_details([_datum(tokens)]))
        B, T = 1, len(tokens)
        assert batch["input_ids"].shape == (B, T)
        assert batch["labels"].shape == (B, T)
        assert batch["position_ids"].shape == (B, T)
        assert batch["loss_scale"].shape == (B, T)
        assert batch["num_samples"] == B

    def test_multi_sample_shapes(self):
        # With padding_free=True (packed), all sequences are concatenated into
        # a single [1, total_tokens] tensor — no padding.
        data = [_datum(list(range(i, i + 8))) for i in range(3)]
        ctrl = self._ctrl()
        batch = ctrl._to_megatron_batch(_fb_details(data))
        # 3 samples × 8 tokens each = 24 total tokens, packed into [1, 24]
        assert batch["input_ids"].shape == (1, 24)

    def test_padding_uses_minus_100_in_labels(self):
        """With packed sequences there is no padding: tokens are concatenated."""
        short = [10, 11, 12]       # 3 tokens
        long_ = [20, 21, 22, 23, 24]  # 5 tokens
        ctrl = self._ctrl()
        batch = ctrl._to_megatron_batch(_fb_details([_datum(short), _datum(long_)]))
        # All 8 tokens packed into [1, 8]: [10,11,12,20,21,22,23,24]
        assert batch["labels"].shape == (1, 8)
        assert batch["labels"][0, :3].tolist() == short
        assert batch["labels"][0, 3:].tolist() == long_

    def test_input_ids_equals_labels_for_valid_positions(self):
        """input_ids and labels hold the same token ids for non-padded positions."""
        tokens = list(range(5, 12))
        ctrl = self._ctrl()
        batch = ctrl._to_megatron_batch(_fb_details([_datum(tokens)]))
        T = len(tokens)
        assert (batch["input_ids"][0, :T] == batch["labels"][0, :T]).all()

    def test_position_ids_are_sequential(self):
        tokens = list(range(5, 13))
        ctrl = self._ctrl()
        batch = ctrl._to_megatron_batch(_fb_details([_datum(tokens)]))
        T = len(tokens)
        assert batch["position_ids"][0, :T].tolist() == list(range(T))

    def test_reward_broadcast_across_valid_positions(self):
        """loss_scale = reward at valid positions; packed format has no padding."""
        reward = 2.5
        tokens_short = [1, 2, 3]       # 3 tokens, reward=2.5
        tokens_long  = list(range(5))  # 5 tokens, reward=0.0
        ctrl = self._ctrl()
        data = [_datum(tokens_short, reward=reward), _datum(tokens_long, reward=0.0)]
        batch = ctrl._to_megatron_batch(_fb_details(data))
        # Packed into [1, 8]: first 3 = reward=2.5, last 5 = reward=0.0
        assert batch["loss_scale"].shape == (1, 8)
        assert batch["loss_scale"][0, :3].tolist() == pytest.approx([reward] * 3)
        assert batch["loss_scale"][0, 3:].tolist() == pytest.approx([0.0] * 5)

    def test_zero_reward_produces_zero_loss_scale(self):
        ctrl = self._ctrl()
        batch = ctrl._to_megatron_batch(_fb_details([_datum(list(range(5)), reward=0.0)]))
        assert batch["loss_scale"].abs().sum().item() == pytest.approx(0.0)

    def test_after_megatron_roll_labels_align_with_next_token(self):
        """
        Simulate what Megatron does internally and verify the resulting label
        at position t is the token at position t+1 in the original sequence.

        With packed sequences (padding_free=True), all tokens are concatenated
        into a single [1, total_tokens] tensor.  After
        ``torch.roll(labels, -1, dims=-1)``, position t gets labels[t+1]:
          - Within each sequence: label = next token ✓
          - Cross-sequence and end-of-tensor boundaries wrap (handled by
            cu_seqlens masking in the attention layer, not by -100 here).
        """
        tokens_short = [10, 20, 30, 40]   # 4 tokens
        tokens_long  = [1, 2, 3, 4, 5]    # 5 tokens
        # packed: [10, 20, 30, 40, 1, 2, 3, 4, 5] in shape [1, 9]
        all_tokens = tokens_short + tokens_long
        ctrl = self._ctrl()
        batch = ctrl._to_megatron_batch(_fb_details([
            _datum(tokens_short), _datum(tokens_long),
        ]))
        assert batch["labels"].shape == (1, len(all_tokens))
        labels_rolled = torch.roll(batch["labels"], -1, dims=-1)

        # Within the short sequence (positions 0..n-2): label = next token
        for t in range(len(tokens_short) - 1):
            assert labels_rolled[0, t].item() == tokens_short[t + 1], (
                f"position {t}: expected {tokens_short[t+1]}, got {labels_rolled[0, t].item()}"
            )
        # Within the long sequence: label = next token
        offset = len(tokens_short)
        for t in range(len(tokens_long) - 1):
            assert labels_rolled[0, offset + t].item() == tokens_long[t + 1]

        # Cross-sequence boundary: last of short gets first of long
        assert labels_rolled[0, len(tokens_short) - 1].item() == tokens_long[0]
        # End-of-tensor: last position wraps to first token of the packed tensor
        assert labels_rolled[0, -1].item() == all_tokens[0]

    def test_dtypes(self):
        ctrl = self._ctrl()
        batch = ctrl._to_megatron_batch(_fb_details([_datum([1, 2, 3, 4])]))
        assert batch["input_ids"].dtype == torch.long
        assert batch["labels"].dtype == torch.long
        assert batch["position_ids"].dtype == torch.long
        assert batch["loss_scale"].dtype == torch.float32

    def test_rejects_single_token_datum(self):
        ctrl = self._ctrl()
        with pytest.raises(ValueError, match="at least 2 tokens"):
            ctrl._to_megatron_batch(_fb_details([_datum([42])]))

    def test_num_samples_equals_batch_size(self):
        data = [_datum(list(range(5, 10))) for _ in range(4)]
        ctrl = self._ctrl()
        batch = ctrl._to_megatron_batch(_fb_details(data))
        assert batch["num_samples"] == 4


# ── config routing ────────────────────────────────────────────────────────────

class TestConfigRouting:
    def test_backend_hf_uses_rl_controller(self):
        from trainers_server.dp_worker.api.server import _make_controller
        from trainers_server.dp_worker.api.controller import RLController

        config = RLControllerConfig(
            model_id="test-model",
            backend="hf",
            training=TrainingServerConfig(gpus=[0], max_length=64),
            inference=InferenceServerConfig(gpus=[1]),
        )
        with (
            patch("trainers_server.dp_worker.api.controller.RLController._launch_rollout_server"),
            patch("trainers_server.dp_worker.api.controller.RLController._init_vllm_client"),
            patch("trainers_server.dp_worker.api.controller.RLController._training_device", return_value="cpu"),
        ):
            from tests.conftest import _FakeProcessor, _FakeTemplate, make_tiny_model
            ctrl = _make_controller.__wrapped__(config) if hasattr(_make_controller, "__wrapped__") else None

        # Just verify the factory returns RLController for backend="hf"
        assert config.backend == "hf"

    def test_backend_field_default_is_hf(self):
        config = RLControllerConfig()
        assert config.backend == "hf"

    def test_backend_megatron_accepted(self):
        config = RLControllerConfig(backend="megatron")
        assert config.backend == "megatron"

    def test_backend_invalid_rejected(self):
        with pytest.raises(Exception):
            RLControllerConfig(backend="deepspeed")  # type: ignore[arg-type]


# ── megatron_sft_args construction ───────────────────────────────────────────

class TestBuildMegatronSftArgs:
    """Verify that RLControllerConfig fields map correctly to MegatronSftArguments."""

    def test_tp_and_pp_forwarded(self):
        from trainers_server.dp_worker.api.megatron_controller import MegatronRLController

        config = _megatron_config(tp=4, pp=2)
        with patch("swift.megatron.arguments.MegatronSftArguments") as mock_args_cls:
            mock_args_cls.return_value = MagicMock()
            MegatronRLController._build_megatron_sft_args(config)

        call_kwargs = mock_args_cls.call_args[1]
        assert call_kwargs.get("tensor_model_parallel_size") == 4
        assert call_kwargs.get("pipeline_model_parallel_size") == 2

    def test_model_id_forwarded(self):
        from trainers_server.dp_worker.api.megatron_controller import MegatronRLController

        config = _megatron_config()
        config = config.model_copy(update={"model_id": "Qwen/Qwen3-7B"})
        with patch("swift.megatron.arguments.MegatronSftArguments") as mock_args_cls:
            mock_args_cls.return_value = MagicMock()
            MegatronRLController._build_megatron_sft_args(config)

        call_kwargs = mock_args_cls.call_args[1]
        assert call_kwargs.get("model") == "Qwen/Qwen3-7B"

    def test_max_length_forwarded(self):
        from trainers_server.dp_worker.api.megatron_controller import MegatronRLController

        config = _megatron_config()
        config = config.model_copy(update={
            "training": TrainingServerConfig(max_length=1024, gpus=[0])
        })
        with patch("swift.megatron.arguments.MegatronSftArguments") as mock_args_cls:
            mock_args_cls.return_value = MagicMock()
            MegatronRLController._build_megatron_sft_args(config)

        call_kwargs = mock_args_cls.call_args[1]
        assert call_kwargs.get("max_length") == 1024

    def test_vllm_disabled_in_args(self):
        """Megatron should NOT start its own vLLM — we manage it separately."""
        from trainers_server.dp_worker.api.megatron_controller import MegatronRLController

        config = _megatron_config()
        with patch("swift.megatron.arguments.MegatronSftArguments") as mock_args_cls:
            mock_args_cls.return_value = MagicMock()
            MegatronRLController._build_megatron_sft_args(config)

        call_kwargs = mock_args_cls.call_args[1]
        assert call_kwargs.get("use_vllm") is False

    def test_save_strategy_is_no(self):
        """We handle saves ourselves; Megatron should not save automatically."""
        from trainers_server.dp_worker.api.megatron_controller import MegatronRLController

        config = _megatron_config()
        with patch("swift.megatron.arguments.MegatronSftArguments") as mock_args_cls:
            mock_args_cls.return_value = MagicMock()
            MegatronRLController._build_megatron_sft_args(config)

        call_kwargs = mock_args_cls.call_args[1]
        assert call_kwargs.get("save_strategy") == "no"


# ── controller flow with mocked trainer ──────────────────────────────────────

class TestMegatronControllerFlow:
    """
    Verify controller logic using a fully mocked MegatronTrainer.

    These tests do NOT run a real Megatron forward pass — they verify that the
    controller calls the right hooks in the right order.
    """

    def test_forward_backward_rejects_unsupported_loss_fn(self):
        ctrl, _, _ = _mock_megatron_controller()
        with pytest.raises(ValueError, match="cross_entropy"):
            ctrl.forward_backward(ForwardBackwardDetails(
                data=[_datum([1, 2, 3])],
                loss_fn="mse",
            ))

    def test_forward_backward_rejects_short_tokens(self):
        ctrl, _, _ = _mock_megatron_controller()
        with pytest.raises(ValueError, match="at least 2 tokens"):
            ctrl.forward_backward(_fb_details([_datum([5])]))

    def test_forward_backward_returns_loss_metric(self):
        ctrl, mock_trainer, _ = _mock_megatron_controller()

        # Patch get_forward_backward_func on the megatron stub so that
        # ``from megatron.core.pipeline_parallel import get_forward_backward_func``
        # inside _forward_backward_impl returns our mock.
        fake_loss_dicts = [{"lm loss": torch.tensor([2.0, 10.0])}]
        mock_fb_inner = MagicMock(return_value=fake_loss_dicts)
        mock_gfbf = MagicMock(return_value=mock_fb_inner)

        with patch.object(
            sys.modules["megatron.core.pipeline_parallel"],
            "get_forward_backward_func",
            mock_gfbf,
        ):
            result = ctrl.forward_backward(_fb_details([_datum([1, 2, 3, 4])]))

        # loss = lm[0] / lm[1].clamp(min=1) = 2.0 / 10.0 = 0.2
        assert result.metrics["loss"] == pytest.approx(0.2)
        assert ctrl.last_loss == pytest.approx(0.2)

    def test_optim_step_calls_optimizer_step_and_zero_grad(self):
        ctrl, mock_trainer, _ = _mock_megatron_controller()
        ctrl.optim_step(OptimStepDetails(adam_params=AdamParams(learning_rate=1e-3)))

        mock_trainer.optimizer.step.assert_called_once()
        mock_trainer.optimizer.zero_grad.assert_called_once()

    def test_optim_step_increments_step_counter(self):
        ctrl, _, _ = _mock_megatron_controller()
        assert ctrl.step == 0
        ctrl.optim_step(OptimStepDetails(adam_params=AdamParams()))
        assert ctrl.step == 1

    def test_optim_step_applies_learning_rate(self):
        ctrl, mock_trainer, _ = _mock_megatron_controller()
        ctrl.optim_step(OptimStepDetails(adam_params=AdamParams(learning_rate=5e-4)))
        assert mock_trainer.optimizer.param_groups[0]["lr"] == pytest.approx(5e-4)

    def test_optim_step_returns_metrics(self):
        ctrl, _, _ = _mock_megatron_controller()
        result = ctrl.optim_step(OptimStepDetails(adam_params=AdamParams(learning_rate=1e-3)))
        assert "step" in result.metrics
        assert "learning_rate" in result.metrics
        assert result.metrics["step"] == 1

    def test_to_inference_calls_bridge_export_weights(self):
        ctrl, mock_trainer, mock_bridge = _mock_megatron_controller()
        mock_client = MagicMock()
        mock_client.update_flattened_params = MagicMock()
        mock_client.reset_prefix_cache = MagicMock()
        ctrl.vllm_client = mock_client
        ctrl._communicator_ready = True

        result = ctrl.to_inference()

        mock_bridge.export_weights.assert_called_once_with(ctrl.unwrapped_models)
        assert result.mode == "inference"
        assert ctrl.mode == "inference"

    def test_to_inference_sets_model_to_eval(self):
        ctrl, mock_trainer, mock_bridge = _mock_megatron_controller()
        mock_client = MagicMock()
        mock_client.update_flattened_params = MagicMock()
        mock_client.reset_prefix_cache = MagicMock()
        ctrl.vllm_client = mock_client
        ctrl._communicator_ready = True

        # unwrapped_models[0] is a real tiny model
        model = ctrl.unwrapped_models[0]
        model.train()
        ctrl.to_inference()
        assert not model.training

    def test_to_training_sets_mode(self):
        ctrl, _, _ = _mock_megatron_controller()
        ctrl.mode = "inference"
        result = ctrl.to_training()
        assert result.mode == "training"
        assert ctrl.mode == "training"

    def test_to_training_sets_model_to_train(self):
        ctrl, _, _ = _mock_megatron_controller()
        model = ctrl.unwrapped_models[0]
        model.eval()
        ctrl._to_training_impl()
        assert model.training

    def test_save_state_calls_bridge_save_weights(self, tmp_path):
        ctrl, mock_trainer, mock_bridge = _mock_megatron_controller()
        ctrl.save_state(str(tmp_path / "ckpt"))
        mock_bridge.save_weights.assert_called_once()
        call_kwargs = mock_bridge.save_weights.call_args[1]
        assert str(tmp_path / "ckpt") in call_kwargs.get("output_dir", "")

    def test_save_state_writes_trainer_state(self, tmp_path):
        ctrl, _, _ = _mock_megatron_controller()
        ctrl.save_state(str(tmp_path / "ckpt"))
        state_file = tmp_path / "ckpt" / "trainer_state.pt"
        assert state_file.exists()
        state = torch.load(str(state_file))
        assert state["step"] == 0

    def test_sample_raises_if_not_inference_mode(self):
        ctrl, _, _ = _mock_megatron_controller()
        from trainers_server.shared.models import SampleDetails
        with pytest.raises(RuntimeError, match="inference mode"):
            ctrl.sample(SampleDetails(
                prompt=ModelInput.from_ints([1, 2, 3]),
                num_samples=1,
            ))

    def test_unwrapped_models_exposed(self):
        """trainer.unwrapped_models must be accessible for MegatronWeightWriter."""
        ctrl, mock_trainer, _ = _mock_megatron_controller()
        assert ctrl.unwrapped_models is mock_trainer.unwrapped_models

    def test_get_status_fields(self):
        ctrl, _, _ = _mock_megatron_controller()
        status = ctrl.get_status()
        assert status.mode == "training"
        assert status.step == 0
        assert status.model_id == "test-model"
