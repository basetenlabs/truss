"""Unit tests for RLController._build_batch.

No GPU, no distributed, no megatron-bridge required.
We exercise only the pure tensor-arithmetic parts: padding, label construction,
loss_mask, position_ids, and reward extraction.
"""

import types
from typing import Optional

import pytest
import torch

from trainers_server.dp_worker.api.controller import RLController
from trainers_server.shared.models import Datum, ModelInput, TensorData


# ── Minimal stand-in that lets us call _build_batch without loading a model ──

class _BatchBuilder:
    """Thin mock controller that satisfies _build_batch's self dependencies."""
    _tokenizer: object = object()  # non-None → _ensure_tokenizer is a no-op
    _pad_token_id: int = 0

    def _ensure_tokenizer(self) -> None:
        pass  # already "loaded"

    # Bind the real implementation as a method on this class.
    _build_batch = RLController._build_batch


def _make_datum(tokens: list[int], reward: float = 1.0) -> Datum:
    return Datum(
        model_input=ModelInput.from_ints(tokens),
        loss_fn_inputs={"reward": TensorData(data=[reward], dtype="float32", shape=[1])},
    )


def _build(data: list[Datum]) -> dict:
    return _BatchBuilder()._build_batch(data)


# ── Shape and type checks ─────────────────────────────────────────────────────


def test_single_sequence_shapes():
    batch = _build([_make_datum([10, 20, 30, 40, 50])])
    B, S = 1, 5
    assert batch["input_ids"].shape == (B, S)
    assert batch["labels"].shape == (B, S)
    assert batch["loss_mask"].shape == (B, S)
    assert batch["position_ids"].shape == (B, S)
    assert batch["rewards"].shape == (B,)


def test_padded_batch_shapes():
    data = [
        _make_datum([1, 2, 3]),         # length 3
        _make_datum([1, 2, 3, 4, 5]),   # length 5 (determines S)
    ]
    batch = _build(data)
    assert batch["input_ids"].shape == (2, 5)
    assert batch["labels"].shape == (2, 5)
    assert batch["loss_mask"].shape == (2, 5)
    assert batch["position_ids"].shape == (2, 5)


def test_dtypes():
    batch = _build([_make_datum([1, 2, 3, 4])])
    assert batch["input_ids"].dtype == torch.long
    assert batch["labels"].dtype == torch.long
    assert batch["loss_mask"].dtype == torch.float
    assert batch["position_ids"].dtype == torch.long
    assert batch["rewards"].dtype == torch.float


# ── input_ids ─────────────────────────────────────────────────────────────────


def test_input_ids_correct():
    tokens = [10, 20, 30, 40]
    batch = _build([_make_datum(tokens)])
    assert batch["input_ids"][0].tolist() == tokens


def test_input_ids_padded_with_zero():
    data = [_make_datum([1, 2]), _make_datum([1, 2, 3, 4])]
    batch = _build(data)
    # Short sequence is padded on the right with pad_token_id (0)
    assert batch["input_ids"][0, 2].item() == 0  # first pad position
    assert batch["input_ids"][0, 3].item() == 0  # second pad position


# ── labels (next-token targets) ───────────────────────────────────────────────


def test_labels_are_next_tokens():
    """labels[i, s] should equal tokens[s+1] for valid positions."""
    tokens = [5, 6, 7, 8, 9]
    batch = _build([_make_datum(tokens)])
    # positions 0..L-2 → next token
    assert batch["labels"][0, 0].item() == 6
    assert batch["labels"][0, 1].item() == 7
    assert batch["labels"][0, 2].item() == 8
    assert batch["labels"][0, 3].item() == 9
    # position L-1 → 0 (masked)
    assert batch["labels"][0, 4].item() == 0


def test_labels_padding_positions_are_zero():
    data = [_make_datum([1, 2]), _make_datum([1, 2, 3, 4])]
    batch = _build(data)
    # For the short sequence, positions 1 and 2,3 should be 0
    # position 0: labels = tokens[1] = 2  ✓
    assert batch["labels"][0, 0].item() == 2
    # position 1 (last valid): masked
    assert batch["labels"][0, 1].item() == 0
    # padding positions
    assert batch["labels"][0, 2].item() == 0
    assert batch["labels"][0, 3].item() == 0


# ── loss_mask ─────────────────────────────────────────────────────────────────


def test_loss_mask_valid_positions():
    """Mask is 1.0 for positions 0..L-2, 0.0 for position L-1 and padding."""
    tokens = [1, 2, 3, 4, 5]  # L=5
    batch = _build([_make_datum(tokens)])
    expected_mask = [1.0, 1.0, 1.0, 1.0, 0.0]  # last position masked
    assert batch["loss_mask"][0].tolist() == expected_mask


def test_loss_mask_with_padding():
    data = [
        _make_datum([10, 20, 30]),        # L=3, S will be 5
        _make_datum([1, 2, 3, 4, 5]),     # L=5
    ]
    batch = _build(data)
    # Row 0: valid positions 0,1 (L-1=2); positions 2,3,4 are 0
    assert batch["loss_mask"][0].tolist() == [1.0, 1.0, 0.0, 0.0, 0.0]
    # Row 1: valid positions 0,1,2,3 (L-1=4); position 4 is 0
    assert batch["loss_mask"][1].tolist() == [1.0, 1.0, 1.0, 1.0, 0.0]


def test_loss_mask_sum_equals_L_minus_1():
    for L in [3, 6, 10]:
        tokens = list(range(L))
        batch = _build([_make_datum(tokens)])
        assert int(batch["loss_mask"][0].sum().item()) == L - 1


# ── position_ids ──────────────────────────────────────────────────────────────


def test_position_ids_valid_range():
    tokens = [1, 2, 3, 4]
    batch = _build([_make_datum(tokens)])
    assert batch["position_ids"][0].tolist() == [0, 1, 2, 3]


def test_position_ids_padding_is_zero():
    data = [_make_datum([1, 2]), _make_datum([1, 2, 3, 4])]
    batch = _build(data)
    # Short sequence: positions 0,1 are 0,1; padding positions are 0
    assert batch["position_ids"][0].tolist() == [0, 1, 0, 0]


# ── rewards ───────────────────────────────────────────────────────────────────


def test_rewards_extracted_correctly():
    data = [
        _make_datum([1, 2, 3], reward=1.0),
        _make_datum([4, 5, 6], reward=0.5),
        _make_datum([7, 8, 9], reward=-1.0),
    ]
    batch = _build(data)
    assert batch["rewards"].tolist() == pytest.approx([1.0, 0.5, -1.0])


def test_missing_reward_defaults_to_one():
    datum = Datum(
        model_input=ModelInput.from_ints([1, 2, 3]),
        loss_fn_inputs={},  # no reward key
    )
    batch = _build([datum])
    assert batch["rewards"][0].item() == pytest.approx(1.0)


def test_nested_list_reward():
    """Reward stored as nested list should be unwrapped to scalar."""
    datum = Datum(
        model_input=ModelInput.from_ints([1, 2, 3]),
        loss_fn_inputs={"reward": TensorData(data=[[0.75]], dtype="float32", shape=[1, 1])},
    )
    batch = _build([datum])
    assert batch["rewards"][0].item() == pytest.approx(0.75)


# ── Error handling ────────────────────────────────────────────────────────────


def test_single_token_raises():
    with pytest.raises(ValueError, match="at least 2 tokens"):
        _build([_make_datum([42])])


def test_empty_batch_raises():
    with pytest.raises(Exception):
        _build([])


# ── Loss math (independent of model) ─────────────────────────────────────────


def test_reward_weighted_loss_formula():
    """Verify the formula: loss = (per_sample_loss * rewards).mean()."""
    # Simulate: 2 samples, S=4
    # per-token losses:
    #   sample 0: [0.4, 0.6, 0.0, 0.0]  mask=[1,1,0,0]
    #   sample 1: [0.2, 0.4, 0.6, 0.0]  mask=[1,1,1,0]
    output_tensor = torch.tensor([[0.4, 0.6, 0.0, 0.0],
                                   [0.2, 0.4, 0.6, 0.0]])
    loss_mask = torch.tensor([[1.0, 1.0, 0.0, 0.0],
                               [1.0, 1.0, 1.0, 0.0]])
    rewards = torch.tensor([1.0, 0.5])

    per_token = output_tensor * loss_mask
    valid_counts = loss_mask.sum(dim=-1).clamp(min=1)
    per_sample = per_token.sum(dim=-1) / valid_counts
    loss = (per_sample * rewards).mean()

    # per_sample[0] = (0.4+0.6)/2 = 0.5,  weighted = 0.5*1.0 = 0.5
    # per_sample[1] = (0.2+0.4+0.6)/3 = 0.4,  weighted = 0.4*0.5 = 0.2
    # mean = (0.5 + 0.2) / 2 = 0.35
    assert loss.item() == pytest.approx(0.35, rel=1e-5)


def test_loss_mask_clamp_prevents_div_zero():
    """A sequence of length 1 would have no valid tokens; clamp(min=1) guards this."""
    output_tensor = torch.tensor([[0.8, 0.0]])
    loss_mask = torch.tensor([[0.0, 0.0]])  # all masked
    rewards = torch.tensor([1.0])

    per_token = output_tensor * loss_mask
    valid_counts = loss_mask.sum(dim=-1).clamp(min=1)
    per_sample = per_token.sum(dim=-1) / valid_counts
    loss = (per_sample * rewards).mean()

    assert loss.item() == pytest.approx(0.0)
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
