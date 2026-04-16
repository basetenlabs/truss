"""
Utilities for multi-rank distributed training (torchrun-compatible).

When launched via ``torchrun --nproc_per_node=N``, the runtime sets:
  RANK, LOCAL_RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT

All public helpers degrade gracefully to single-process (rank=0, world=1)
when torch.distributed is not initialized, so callers never need an
``if is_distributed()`` guard for basic queries.
"""
from __future__ import annotations

import enum
import logging
import os
from typing import Any

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Process-group lifecycle
# ---------------------------------------------------------------------------

def init_process_group(backend: str = "nccl") -> None:
    """Initialize the default process group when running under torchrun.

    No-ops when ``WORLD_SIZE`` is unset or 1 (single-process run).
    """
    if not dist.is_available():
        return
    if dist.is_initialized():
        return
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return
    dist.init_process_group(backend=backend)
    logger.info(
        "torch.distributed initialized: rank=%d/%d backend=%s local_rank=%d",
        dist.get_rank(),
        dist.get_world_size(),
        backend,
        get_local_rank(),
    )


def destroy_process_group() -> None:
    """Tear down the process group if it was initialized."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Rank / world-size helpers
# ---------------------------------------------------------------------------

def is_distributed() -> bool:
    """Return True iff torch.distributed is initialized with world_size > 1."""
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def get_local_rank() -> int:
    """Local rank within the current node (set by torchrun as LOCAL_RANK)."""
    return int(os.environ.get("LOCAL_RANK", "0"))


def is_rank_zero() -> bool:
    return get_rank() == 0


# ---------------------------------------------------------------------------
# Collective helpers
# ---------------------------------------------------------------------------

def barrier() -> None:
    if is_distributed():
        dist.barrier()


def broadcast_object(obj: Any, src: int = 0) -> Any:
    """Broadcast an arbitrary picklable object from *src* to all ranks.

    On the source rank, *obj* is the value to send.
    On all other ranks, *obj* is ignored and the broadcast value is returned.
    """
    if not is_distributed():
        return obj
    buf: list[Any] = [obj]
    dist.broadcast_object_list(buf, src=src)
    return buf[0]


# ---------------------------------------------------------------------------
# Worker operation codes
# ---------------------------------------------------------------------------

class WorkerOp(enum.IntEnum):
    """Op codes broadcast from rank-0 to worker ranks."""

    SHUTDOWN = 0
    FORWARD_BACKWARD = 1
    OPTIM_STEP = 2
    TO_INFERENCE = 3
    TO_TRAINING = 4
    SAVE_STATE = 5
