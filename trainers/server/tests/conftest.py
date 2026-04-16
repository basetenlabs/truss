"""
Shared test fixtures.

All tests run on CPU without any GPU or ms-swift installed.  We patch the
three ms-swift callsites in controller.py at import time so the module can
be imported cleanly, then provide a pre-built ``RLController`` that uses a
tiny two-layer transformer (CPU).
"""
from __future__ import annotations

import sys
import types
from typing import Any, List, Optional
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Tiny model / processor stubs
# ---------------------------------------------------------------------------

class _TinyTransformerBlock(nn.Module):
    """A single-layer transformer block used as a stand-in for real models."""

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.linear = nn.Linear(d_model, vocab_size)

    def save_pretrained(self, path):  # HF-compatible stub
        pass

    def forward(self, input_ids=None, attention_mask=None, **_kwargs):
        x = self.embed(input_ids)          # (B, T, d)
        logits = self.linear(x)            # (B, T, vocab_size)

        class _FakeOutput:
            pass

        out = _FakeOutput()
        out.logits = logits
        return out


VOCAB_SIZE = 128
D_MODEL = 32


def make_tiny_model() -> nn.Module:
    return _TinyTransformerBlock(D_MODEL, VOCAB_SIZE)


class _FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def decode(self, tokens, skip_special_tokens=False):  # noqa: ARG002
        return " ".join(str(t) for t in tokens)

    def encode(self, text, add_special_tokens=False):  # noqa: ARG002
        return [int(w) % VOCAB_SIZE for w in text.split() if w.isdigit()]


class _FakeProcessor:
    tokenizer = _FakeTokenizer()

    def save_pretrained(self, path):
        pass


class _FakeTemplate:
    def set_mode(self, mode):
        pass


# ---------------------------------------------------------------------------
# ms-swift mock setup
#
# controller.py imports from swift.*; we install lightweight stub modules so
# the import succeeds even without ms-swift installed.
# ---------------------------------------------------------------------------

def _install_swift_stubs() -> None:
    """Register stub swift.* modules in sys.modules (idempotent).

    Skipped when the real ms-swift package is already importable — this lets
    smoke tests that have the full worker extras installed use the real library.
    """
    if "swift" in sys.modules:
        return
    # Don't stub if the real swift is importable (e.g. on a GPU node with
    # worker extras installed).
    try:
        import importlib.util
        if importlib.util.find_spec("swift") is not None:
            return
    except Exception:
        pass

    swift = types.ModuleType("swift")
    sys.modules["swift"] = swift

    for sub in [
        "swift.arguments",
        "swift.infer_engine",
        "swift.infer_engine.protocol",
        "swift.model",
        "swift.model.register",
        "swift.pipelines",
        "swift.pipelines.infer",
        "swift.pipelines.infer.rollout",
        "swift.rlhf_trainers",
        "swift.rlhf_trainers.utils",
        "swift.rlhf_trainers.vllm_client",
        "swift.template",
        "swift.template.register",
    ]:
        mod = types.ModuleType(sub)
        sys.modules[sub] = mod

    # RolloutArguments
    sys.modules["swift.arguments"].RolloutArguments = MagicMock()

    # RequestConfig / RolloutInferRequest
    sys.modules["swift.infer_engine"].RequestConfig = MagicMock()
    sys.modules["swift.infer_engine.protocol"].RolloutInferRequest = MagicMock()

    # get_model_processor — returns (tiny_model, fake_processor)
    def _fake_get_model_processor(model_id, **kwargs):
        return make_tiny_model(), _FakeProcessor()

    sys.modules["swift.model.register"].get_model_processor = _fake_get_model_processor

    # rollout_main — no-op
    sys.modules["swift.pipelines.infer.rollout"].rollout_main = MagicMock()

    # FlattenedTensorBucket
    class _FakeFlattenedTensorBucket:
        def __init__(self, named_tensors):
            self._named_tensors = named_tensors

        def get_metadata(self):
            return [(name, tuple(t.shape), str(t.dtype)) for name, t in self._named_tensors]

        def get_flattened_tensor(self):
            return torch.cat([t.flatten() for _, t in self._named_tensors])

    sys.modules["swift.rlhf_trainers.utils"].FlattenedTensorBucket = _FakeFlattenedTensorBucket

    # VLLMClient
    sys.modules["swift.rlhf_trainers.vllm_client"].VLLMClient = MagicMock()

    # get_template
    sys.modules["swift.template.register"].get_template = lambda *a, **kw: _FakeTemplate()


_install_swift_stubs()


# ---------------------------------------------------------------------------
# Controller factory fixture
# ---------------------------------------------------------------------------

def _make_controller_no_rollout(model=None, processor=None) -> "RLController":
    """Build an RLController that skips rollout server startup."""
    from trainers_server.dp_worker.api.models import RLControllerConfig, TrainingServerConfig, InferenceServerConfig
    from trainers_server.dp_worker.api.controller import RLController

    config = RLControllerConfig(
        model_id="test-model",
        training=TrainingServerConfig(gpus=[0], max_length=64),
        inference=InferenceServerConfig(gpus=[1]),
    )

    if model is None:
        model = make_tiny_model()
    if processor is None:
        processor = _FakeProcessor()

    with (
        patch("trainers_server.dp_worker.api.controller.RLController._launch_rollout_server"),
        patch("trainers_server.dp_worker.api.controller.RLController._init_vllm_client"),
        patch("trainers_server.dp_worker.api.controller.RLController._training_device", return_value="cpu"),
    ):
        ctrl = RLController(config, model=model, processor=processor, template=_FakeTemplate())

    # Ensure the model is on CPU for tests.
    ctrl.model = ctrl.model.to("cpu")
    ctrl.optimizer = torch.optim.AdamW(ctrl.model.parameters(), lr=0.0)
    ctrl.optimizer.zero_grad(set_to_none=True)
    return ctrl


@pytest.fixture()
def controller():
    return _make_controller_no_rollout()
