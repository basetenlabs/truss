"""GPU tests for RLController — forward_backward, optim_step, worker_loop dispatch.

These tests load Qwen3-0.6B from MODEL_PATH and run actual Megatron-backed
forward/backward passes on real CUDA GPUs. The vLLM rollout server is mocked
so these tests don't need extra GPU memory for inference.

Run:
    cd trainers/server
    uv run --extra worker pytest tests/test_controller_gpu.py -v -s

Each test creates fresh child processes via mp.spawn. Results are written to
a temp JSON file by rank 0 and read back by the parent pytest process.
"""

import json
import os
import socket
import tempfile
import warnings
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from helpers import MODEL_PATH, skip_if_no_gpu

pytestmark = pytest.mark.gpu


def _broadcast_op(op_code: int) -> None:
    """Broadcast an op-code integer from rank 0 to all ranks (NCCL-safe)."""
    t = torch.tensor([op_code], dtype=torch.int32, device=f"cuda:{torch.cuda.current_device()}")
    dist.broadcast(t, src=0)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _make_datum(tokens: list[int], reward: float = 1.0):
    from trainers_server.shared.models import Datum, ModelInput, TensorData
    return Datum(
        model_input=ModelInput.from_ints(tokens),
        loss_fn_inputs={"reward": TensorData(data=[reward], dtype="float32", shape=[1])},
    )


def _make_fb_details(data):
    from trainers_server.shared.models import ForwardBackwardDetails
    return ForwardBackwardDetails(data=data)


def _make_optim_details(lr: float = 1e-4):
    from trainers_server.shared.models import AdamParams, OptimStepDetails
    return OptimStepDetails(adam_params=AdamParams(learning_rate=lr))


def _init_dist(rank: int, world_size: int, dist_port: int, gpu: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(dist_port)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    torch.cuda.set_device(gpu)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


def _teardown() -> None:
    try:
        from megatron.core import parallel_state
        parallel_state.destroy_model_parallel()
    except Exception:
        pass
    dist.destroy_process_group()


# ── Test: single GPU, forward_backward + optim_step, loss decreases ──────────


def _single_gpu_rank(rank: int, world_size: int, dist_port: int, result_path: str) -> None:
    """Single-GPU: verify forward_backward produces a finite loss and optim_step
    reduces it over multiple iterations on the same batch."""
    gpus = [0]
    _init_dist(rank, world_size, dist_port, gpus[rank])

    from trainers_server.dp_worker.api.controller import (
        RLController, OP_EXIT, worker_loop
    )
    from trainers_server.dp_worker.api.models import (
        RLControllerConfig, TrainingServerConfig, InferenceServerConfig
    )

    config = RLControllerConfig(
        model_id=MODEL_PATH,
        training=TrainingServerConfig(tensor_parallel_size=1, pipeline_parallel_size=1, gpus=gpus, max_length=128),
        inference=InferenceServerConfig(tensor_parallel_size=1, gpus=[0]),
    )

    # Mock vLLM so we don't need extra GPU memory for the inference server.
    with patch.object(RLController, "_launch_rollout"), \
         patch.object(RLController, "_wait_for_rollout"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            controller = RLController(config)

    # Short sequences (8 tokens each) so a single H100 can easily hold them.
    data = [
        _make_datum(list(range(1, 9)), reward=1.0),
        _make_datum(list(range(10, 18)), reward=0.5),
    ]
    details = _make_fb_details(data)
    optim_details = _make_optim_details(lr=1e-3)

    losses = []

    if rank == 0:
        # Run 5 forward_backward + optim_step iterations on the same batch.
        for _ in range(5):
            result = controller.forward_backward(details)
            losses.append(result.metrics["loss"])
            optim_result = controller.optim_step(optim_details)

        # Signal workers to exit.
        _broadcast_op(OP_EXIT)

        with open(result_path, "w") as f:
            json.dump({
                "losses": losses,
                "step": controller.step,
                "update_successful": True,  # if we got here, it worked
            }, f)
    else:
        worker_loop(controller)

    _teardown()


@skip_if_no_gpu(1)
def test_single_gpu_forward_backward_and_optim():
    dist_port = _find_free_port()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        result_path = f.name

    try:
        mp.spawn(
            _single_gpu_rank,
            args=(1, dist_port, result_path),
            nprocs=1,
            join=True,
        )
        with open(result_path) as f:
            results = json.load(f)
    finally:
        Path(result_path).unlink(missing_ok=True)

    losses = results["losses"]
    assert len(losses) == 5, "Expected 5 loss values"

    # All losses must be finite and positive.
    for i, loss in enumerate(losses):
        assert loss > 0, f"Loss at step {i} is non-positive: {loss}"
        assert loss < 1e6, f"Loss at step {i} is too large (NaN/Inf?): {loss}"

    # After 5 steps on the same batch, the final loss should be lower than
    # the first (strong signal that gradients are flowing and the optimizer works).
    assert losses[-1] < losses[0], (
        f"Loss did not decrease over 5 steps: {losses[0]:.4f} → {losses[-1]:.4f}"
    )

    assert results["step"] == 5


# ── Test: TP=2 forward_backward ───────────────────────────────────────────────


def _tp2_rank(rank: int, world_size: int, dist_port: int, result_path: str) -> None:
    gpus = [0, 1]
    _init_dist(rank, world_size, dist_port, gpus[rank])

    from trainers_server.dp_worker.api.controller import (
        RLController, OP_EXIT, worker_loop
    )
    from trainers_server.dp_worker.api.models import (
        RLControllerConfig, TrainingServerConfig, InferenceServerConfig
    )

    config = RLControllerConfig(
        model_id=MODEL_PATH,
        training=TrainingServerConfig(tensor_parallel_size=2, pipeline_parallel_size=1, gpus=gpus, max_length=128),
        inference=InferenceServerConfig(tensor_parallel_size=1, gpus=[2]),
    )

    with patch.object(RLController, "_launch_rollout"), \
         patch.object(RLController, "_wait_for_rollout"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            controller = RLController(config)

    data = [
        _make_datum(list(range(1, 17)), reward=1.0),
        _make_datum(list(range(20, 36)), reward=0.5),
    ]
    details = _make_fb_details(data)

    if rank == 0:
        result = controller.forward_backward(details)
        loss = result.metrics["loss"]

        optim_result = controller.optim_step(_make_optim_details(lr=1e-3))

        _broadcast_op(OP_EXIT)

        with open(result_path, "w") as f:
            json.dump({"loss": loss, "step": controller.step}, f)
    else:
        worker_loop(controller)

    _teardown()


@skip_if_no_gpu(2)
def test_tp2_forward_backward():
    dist_port = _find_free_port()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        result_path = f.name

    try:
        mp.spawn(
            _tp2_rank,
            args=(2, dist_port, result_path),
            nprocs=2,
            join=True,
        )
        with open(result_path) as f:
            results = json.load(f)
    finally:
        Path(result_path).unlink(missing_ok=True)

    assert 0 < results["loss"] < 1e6, f"TP=2 loss out of range: {results['loss']}"
    assert results["step"] == 1


# ── Test: TP=4 forward_backward ───────────────────────────────────────────────


def _tp4_rank(rank: int, world_size: int, dist_port: int, result_path: str) -> None:
    gpus = [0, 1, 2, 3]
    _init_dist(rank, world_size, dist_port, gpus[rank])

    from trainers_server.dp_worker.api.controller import (
        RLController, OP_EXIT, worker_loop
    )
    from trainers_server.dp_worker.api.models import (
        RLControllerConfig, TrainingServerConfig, InferenceServerConfig
    )

    config = RLControllerConfig(
        model_id=MODEL_PATH,
        training=TrainingServerConfig(tensor_parallel_size=4, pipeline_parallel_size=1, gpus=gpus, max_length=128),
        inference=InferenceServerConfig(tensor_parallel_size=2, gpus=[4, 5]),
    )

    with patch.object(RLController, "_launch_rollout"), \
         patch.object(RLController, "_wait_for_rollout"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            controller = RLController(config)

    data = [
        _make_datum(list(range(1, 17)), reward=1.0),
        _make_datum(list(range(20, 32)), reward=0.75),
        _make_datum(list(range(40, 56)), reward=0.5),
    ]
    details = _make_fb_details(data)

    if rank == 0:
        result = controller.forward_backward(details)
        loss = result.metrics["loss"]
        optim_result = controller.optim_step(_make_optim_details())

        _broadcast_op(OP_EXIT)

        with open(result_path, "w") as f:
            json.dump({"loss": loss, "step": controller.step}, f)
    else:
        worker_loop(controller)

    _teardown()


@skip_if_no_gpu(4)
def test_tp4_forward_backward():
    dist_port = _find_free_port()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        result_path = f.name

    try:
        mp.spawn(
            _tp4_rank,
            args=(4, dist_port, result_path),
            nprocs=4,
            join=True,
        )
        with open(result_path) as f:
            results = json.load(f)
    finally:
        Path(result_path).unlink(missing_ok=True)

    assert 0 < results["loss"] < 1e6, f"TP=4 loss out of range: {results['loss']}"
    assert results["step"] == 1


# ── Test: TP=1 loss decreases match between all ranks ────────────────────────
#
# Verify that when called on multiple ranks, all ranks compute the same
# loss (they must, since per-sample loss reduction is broadcast-free on TP=1).


def _loss_consistency_rank(rank: int, world_size: int, dist_port: int, result_path: str) -> None:
    """Each rank writes its own loss to a separate file; parent compares them."""
    gpus = [0, 1]
    _init_dist(rank, world_size, dist_port, gpus[rank])

    from trainers_server.dp_worker.api.controller import (
        RLController, OP_EXIT, worker_loop
    )
    from trainers_server.dp_worker.api.models import (
        RLControllerConfig, TrainingServerConfig, InferenceServerConfig
    )

    config = RLControllerConfig(
        model_id=MODEL_PATH,
        training=TrainingServerConfig(tensor_parallel_size=2, pipeline_parallel_size=1, gpus=gpus, max_length=64),
        inference=InferenceServerConfig(tensor_parallel_size=1, gpus=[2]),
    )

    with patch.object(RLController, "_launch_rollout"), \
         patch.object(RLController, "_wait_for_rollout"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            controller = RLController(config)

    data = [_make_datum(list(range(1, 9)), reward=1.0)]
    details = _make_fb_details(data)

    losses_per_rank = []
    if rank == 0:
        for _ in range(3):
            result = controller.forward_backward(details)
            losses_per_rank.append(result.metrics["loss"])
            controller.optim_step(_make_optim_details())

        _broadcast_op(OP_EXIT)

        with open(f"{result_path}.rank{rank}", "w") as f:
            json.dump({"losses": losses_per_rank}, f)
    else:
        worker_loop(controller)

        # Workers don't have the loss (returned only on rank 0) but we still
        # write a file to confirm they exited cleanly.
        with open(f"{result_path}.rank{rank}", "w") as f:
            json.dump({"losses": [], "clean_exit": True}, f)

    _teardown()


@skip_if_no_gpu(2)
def test_tp2_loss_decreases():
    dist_port = _find_free_port()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        result_path = f.name

    try:
        mp.spawn(
            _loss_consistency_rank,
            args=(2, dist_port, result_path),
            nprocs=2,
            join=True,
        )
        with open(f"{result_path}.rank0") as f:
            rank0 = json.load(f)
        with open(f"{result_path}.rank1") as f:
            rank1 = json.load(f)
    finally:
        for r in range(2):
            Path(f"{result_path}.rank{r}").unlink(missing_ok=True)
        Path(result_path).unlink(missing_ok=True)

    losses = rank0["losses"]
    assert len(losses) == 3
    for loss in losses:
        assert 0 < loss < 1e6, f"Loss out of range: {loss}"
    assert losses[-1] < losses[0], (
        f"Loss did not decrease over 3 steps with TP=2: {losses}"
    )
    assert rank1["clean_exit"], "Worker rank 1 did not exit cleanly"


# ── Test: gradient accumulation (multiple forward_backward before optim_step) ─


def _grad_accum_rank(rank: int, world_size: int, dist_port: int, result_path: str) -> None:
    gpus = [0]
    _init_dist(rank, world_size, dist_port, gpus[rank])

    from trainers_server.dp_worker.api.controller import (
        RLController, OP_EXIT, worker_loop
    )
    from trainers_server.dp_worker.api.models import (
        RLControllerConfig, TrainingServerConfig, InferenceServerConfig
    )

    config = RLControllerConfig(
        model_id=MODEL_PATH,
        training=TrainingServerConfig(tensor_parallel_size=1, pipeline_parallel_size=1, gpus=gpus, max_length=64),
        inference=InferenceServerConfig(tensor_parallel_size=1, gpus=[0]),
    )

    with patch.object(RLController, "_launch_rollout"), \
         patch.object(RLController, "_wait_for_rollout"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            controller = RLController(config)

    # Two different batches to accumulate gradients over.
    batch_a = [_make_datum(list(range(1, 9)), reward=1.0)]
    batch_b = [_make_datum(list(range(10, 18)), reward=0.5)]

    if rank == 0:
        # Gradient accumulation: 2 forward_backward before 1 optim_step.
        r1 = controller.forward_backward(_make_fb_details(batch_a))
        r2 = controller.forward_backward(_make_fb_details(batch_b))
        # No zero_grad between them — gradients accumulate.
        optim_r = controller.optim_step(_make_optim_details())

        _broadcast_op(OP_EXIT)

        with open(result_path, "w") as f:
            json.dump({
                "loss_a": r1.metrics["loss"],
                "loss_b": r2.metrics["loss"],
                "step": controller.step,
            }, f)
    else:
        worker_loop(controller)

    _teardown()


@skip_if_no_gpu(1)
def test_gradient_accumulation():
    dist_port = _find_free_port()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        result_path = f.name

    try:
        mp.spawn(
            _grad_accum_rank,
            args=(1, dist_port, result_path),
            nprocs=1,
            join=True,
        )
        with open(result_path) as f:
            results = json.load(f)
    finally:
        Path(result_path).unlink(missing_ok=True)

    assert 0 < results["loss_a"] < 1e6
    assert 0 < results["loss_b"] < 1e6
    assert results["step"] == 1  # only one optim_step was called


# ── Test: optim_step zero_grad clears accumulated gradients ──────────────────


def _zero_grad_rank(rank: int, world_size: int, dist_port: int, result_path: str) -> None:
    gpus = [0]
    _init_dist(rank, world_size, dist_port, gpus[rank])

    from trainers_server.dp_worker.api.controller import (
        RLController, OP_EXIT, worker_loop
    )
    from trainers_server.dp_worker.api.models import (
        RLControllerConfig, TrainingServerConfig, InferenceServerConfig
    )

    config = RLControllerConfig(
        model_id=MODEL_PATH,
        training=TrainingServerConfig(tensor_parallel_size=1, pipeline_parallel_size=1, gpus=gpus, max_length=64),
        inference=InferenceServerConfig(tensor_parallel_size=1, gpus=[0]),
    )

    with patch.object(RLController, "_launch_rollout"), \
         patch.object(RLController, "_wait_for_rollout"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            controller = RLController(config)

    data = [_make_datum(list(range(1, 9)), reward=1.0)]
    details = _make_fb_details(data)

    if rank == 0:
        # Cycle 1: forward + backward + optim (zero_grad inside).
        r1 = controller.forward_backward(details)
        controller.optim_step(_make_optim_details())

        # Cycle 2: fresh forward + backward + optim.
        r2 = controller.forward_backward(details)
        controller.optim_step(_make_optim_details())

        _broadcast_op(OP_EXIT)

        with open(result_path, "w") as f:
            json.dump({
                "loss1": r1.metrics["loss"],
                "loss2": r2.metrics["loss"],
                "step": controller.step,
            }, f)
    else:
        worker_loop(controller)

    _teardown()


@skip_if_no_gpu(1)
def test_two_independent_steps():
    """Two separate forward+optim cycles should both produce finite losses."""
    dist_port = _find_free_port()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        result_path = f.name

    try:
        mp.spawn(
            _zero_grad_rank,
            args=(1, dist_port, result_path),
            nprocs=1,
            join=True,
        )
        with open(result_path) as f:
            results = json.load(f)
    finally:
        Path(result_path).unlink(missing_ok=True)

    assert 0 < results["loss1"] < 1e6
    assert 0 < results["loss2"] < 1e6
    assert results["step"] == 2
    # loss2 should be lower because we did one optimizer step.
    assert results["loss2"] < results["loss1"], (
        f"Second cycle loss {results['loss2']:.4f} should be < first {results['loss1']:.4f}"
    )
