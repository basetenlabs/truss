"""Tests for the distributed broadcast helpers using the CPU gloo backend.

These tests verify _broadcast_bytes and _broadcast_tensor without requiring
CUDA. They use mp.spawn to create a small process group.
"""

import os
import pickle
import socket
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


# ── Rank worker functions ──────────────────────────────────────────────────────


def _setup_gloo(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)


def _teardown() -> None:
    dist.destroy_process_group()


# ── _broadcast_bytes ──────────────────────────────────────────────────────────


def _test_broadcast_bytes_rank(rank: int, world_size: int, port: int, result_path: str) -> None:
    """Worker: rank 0 sends a payload; all ranks write their received value."""
    _setup_gloo(rank, world_size, port)

    # Import after dist is initialized so _is_distributed() returns True.
    from trainers_server.dp_worker.api.controller import _broadcast_bytes

    if rank == 0:
        payload = pickle.dumps({"hello": "world", "rank": 0})
        received = _broadcast_bytes(payload)
    else:
        received = _broadcast_bytes(None)

    # Each process writes its own result file.
    with open(f"{result_path}.rank{rank}", "wb") as f:
        f.write(received)

    _teardown()


@pytest.mark.parametrize("world_size", [2, 3])
def test_broadcast_bytes(tmp_path, world_size):
    port = _find_free_port()
    result_base = str(tmp_path / "result")

    mp.spawn(
        _test_broadcast_bytes_rank,
        args=(world_size, port, result_base),
        nprocs=world_size,
        join=True,
    )

    expected = pickle.dumps({"hello": "world", "rank": 0})
    for r in range(world_size):
        with open(f"{result_base}.rank{r}", "rb") as f:
            received = f.read()
        assert received == expected, f"rank {r} got wrong bytes"


def _test_broadcast_large_bytes_rank(rank: int, world_size: int, port: int, result_path: str) -> None:
    _setup_gloo(rank, world_size, port)
    from trainers_server.dp_worker.api.controller import _broadcast_bytes

    large_obj = {"data": list(range(10_000)), "extra": "x" * 1000}
    if rank == 0:
        payload = pickle.dumps(large_obj)
        received = _broadcast_bytes(payload)
    else:
        received = _broadcast_bytes(None)

    decoded = pickle.loads(received)
    with open(f"{result_path}.rank{rank}", "w") as f:
        f.write(str(len(decoded["data"])))

    _teardown()


def test_broadcast_large_bytes(tmp_path):
    port = _find_free_port()
    result_base = str(tmp_path / "large")

    mp.spawn(
        _test_broadcast_large_bytes_rank,
        args=(2, port, result_base),
        nprocs=2,
        join=True,
    )

    for r in range(2):
        with open(f"{result_base}.rank{r}") as f:
            count = int(f.read())
        assert count == 10_000, f"rank {r}: large broadcast truncated"


# ── _broadcast_tensor ─────────────────────────────────────────────────────────


def _test_broadcast_tensor_rank(rank: int, world_size: int, port: int, result_path: str) -> None:
    _setup_gloo(rank, world_size, port)
    from trainers_server.dp_worker.api.controller import _broadcast_tensor

    if rank == 0:
        t = torch.tensor([42, 99, -7], dtype=torch.int64)
    else:
        t = torch.zeros(3, dtype=torch.int64)

    _broadcast_tensor(t)

    with open(f"{result_path}.rank{rank}", "w") as f:
        f.write(",".join(str(x) for x in t.tolist()))

    _teardown()


def test_broadcast_tensor(tmp_path):
    port = _find_free_port()
    result_base = str(tmp_path / "tensor")

    mp.spawn(
        _test_broadcast_tensor_rank,
        args=(2, port, result_base),
        nprocs=2,
        join=True,
    )

    for r in range(2):
        with open(f"{result_base}.rank{r}") as f:
            values = [int(x) for x in f.read().split(",")]
        assert values == [42, 99, -7], f"rank {r} got wrong tensor"


# ── No-op when not distributed ────────────────────────────────────────────────


def test_broadcast_bytes_noop_without_dist():
    """When dist is not initialized, _broadcast_bytes is a no-op passthrough."""
    from trainers_server.dp_worker.api.controller import _broadcast_bytes

    # Ensure dist is NOT initialized for this test (single-process, no setup).
    assert not dist.is_initialized()

    data = b"hello"
    result = _broadcast_bytes(data)
    assert result == data


def test_broadcast_tensor_noop_without_dist():
    """When dist is not initialized, _broadcast_tensor is a no-op."""
    from trainers_server.dp_worker.api.controller import _broadcast_tensor

    assert not dist.is_initialized()

    t = torch.tensor([1, 2, 3])
    _broadcast_tensor(t)  # should not raise
    assert t.tolist() == [1, 2, 3]


# ── op-code broadcast integrity ───────────────────────────────────────────────


def _test_opcode_rank(rank: int, world_size: int, port: int, result_path: str) -> None:
    """Verify op-code broadcast matches the op-code constants in controller."""
    _setup_gloo(rank, world_size, port)
    from trainers_server.dp_worker.api.controller import (
        OP_FORWARD_BACKWARD,
        OP_OPTIM_STEP,
        OP_TO_INFERENCE,
        OP_EXIT,
    )

    received_ops = []
    for expected_op in [OP_FORWARD_BACKWARD, OP_OPTIM_STEP, OP_TO_INFERENCE, OP_EXIT]:
        op_t = torch.zeros(1, dtype=torch.int32)
        if rank == 0:
            op_t[0] = expected_op
        dist.broadcast(op_t, src=0)
        received_ops.append(int(op_t.item()))

    with open(f"{result_path}.rank{rank}", "w") as f:
        f.write(",".join(str(x) for x in received_ops))

    _teardown()


def test_opcode_broadcast(tmp_path):
    port = _find_free_port()
    result_base = str(tmp_path / "opcode")

    mp.spawn(
        _test_opcode_rank,
        args=(2, port, result_base),
        nprocs=2,
        join=True,
    )

    from trainers_server.dp_worker.api.controller import (
        OP_FORWARD_BACKWARD, OP_OPTIM_STEP, OP_TO_INFERENCE, OP_EXIT
    )
    expected = [OP_FORWARD_BACKWARD, OP_OPTIM_STEP, OP_TO_INFERENCE, OP_EXIT]

    for r in range(2):
        with open(f"{result_base}.rank{r}") as f:
            ops = [int(x) for x in f.read().split(",")]
        assert ops == expected, f"rank {r} received wrong op codes"
