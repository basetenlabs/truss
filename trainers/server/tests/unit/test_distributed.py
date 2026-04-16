"""Unit tests for trainers_server.dp_worker.distributed."""
from __future__ import annotations

import os
from unittest.mock import patch, MagicMock

import pytest
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _import_fresh():
    """Re-import distributed with a clean torch.distributed mock."""
    import importlib
    import trainers_server.dp_worker.distributed as dist_mod
    importlib.reload(dist_mod)
    return dist_mod


# ---------------------------------------------------------------------------
# Single-rank (no torch.distributed) behaviour
# ---------------------------------------------------------------------------

class TestSingleRank:
    def test_is_distributed_false_when_not_initialized(self):
        from trainers_server.dp_worker.distributed import is_distributed
        assert is_distributed() is False

    def test_get_rank_returns_zero(self):
        from trainers_server.dp_worker.distributed import get_rank
        assert get_rank() == 0

    def test_get_world_size_returns_one(self):
        from trainers_server.dp_worker.distributed import get_world_size
        assert get_world_size() == 1

    def test_is_rank_zero_true(self):
        from trainers_server.dp_worker.distributed import is_rank_zero
        assert is_rank_zero() is True

    def test_get_local_rank_defaults_zero(self):
        from trainers_server.dp_worker.distributed import get_local_rank
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LOCAL_RANK", None)
            assert get_local_rank() == 0

    def test_get_local_rank_reads_env(self):
        from trainers_server.dp_worker.distributed import get_local_rank
        with patch.dict(os.environ, {"LOCAL_RANK": "3"}):
            assert get_local_rank() == 3

    def test_broadcast_object_identity_when_not_distributed(self):
        from trainers_server.dp_worker.distributed import broadcast_object
        obj = {"key": [1, 2, 3]}
        result = broadcast_object(obj, src=0)
        assert result is obj

    def test_barrier_noop_when_not_distributed(self):
        from trainers_server.dp_worker.distributed import barrier
        # Should not raise.
        barrier()

    def test_init_process_group_noop_when_world_size_one(self):
        from trainers_server.dp_worker.distributed import init_process_group
        import torch.distributed as dist
        with patch.dict(os.environ, {"WORLD_SIZE": "1"}):
            init_process_group()
        assert not dist.is_initialized()

    def test_init_process_group_noop_when_env_absent(self):
        from trainers_server.dp_worker.distributed import init_process_group
        import torch.distributed as dist
        env = {k: v for k, v in os.environ.items() if k != "WORLD_SIZE"}
        with patch.dict(os.environ, env, clear=True):
            init_process_group()
        assert not dist.is_initialized()


# ---------------------------------------------------------------------------
# WorkerOp enum
# ---------------------------------------------------------------------------

class TestWorkerOp:
    def test_all_ops_defined(self):
        from trainers_server.dp_worker.distributed import WorkerOp
        names = {op.name for op in WorkerOp}
        assert names == {"SHUTDOWN", "FORWARD_BACKWARD", "OPTIM_STEP", "TO_INFERENCE", "TO_TRAINING", "SAVE_STATE"}

    def test_shutdown_is_zero(self):
        from trainers_server.dp_worker.distributed import WorkerOp
        assert WorkerOp.SHUTDOWN == 0

    def test_roundtrip_from_int(self):
        from trainers_server.dp_worker.distributed import WorkerOp
        for op in WorkerOp:
            assert WorkerOp(int(op)) == op


# ---------------------------------------------------------------------------
# broadcast_object with mocked torch.distributed
# ---------------------------------------------------------------------------

class TestBroadcastObject:
    def _make_dist_mock(self):
        mock_dist = MagicMock()
        mock_dist.is_available.return_value = True
        mock_dist.is_initialized.return_value = True
        mock_dist.get_world_size.return_value = 4
        mock_dist.get_rank.return_value = 0

        def fake_broadcast_object_list(buf, src=0):
            # Simulate: buf[0] already contains rank-0 value; other ranks would
            # receive it.  In tests we just leave buf unchanged (rank 0 path).
            pass

        mock_dist.broadcast_object_list.side_effect = fake_broadcast_object_list
        return mock_dist

    def test_broadcast_calls_dist_when_distributed(self):
        import trainers_server.dp_worker.distributed as dist_mod
        mock_dist = self._make_dist_mock()
        with patch.object(dist_mod, "dist", mock_dist):
            obj = {"hello": "world"}
            result = dist_mod.broadcast_object(obj, src=0)
        mock_dist.broadcast_object_list.assert_called_once()
        # Rank 0: returned value is the object we put in the buffer.
        assert result == obj
