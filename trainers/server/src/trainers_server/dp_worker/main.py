"""Entry point for the dp_worker server.

Single-GPU (TP=PP=1):
    python -m trainers_server.dp_worker.main --config config.json

Multi-GPU (e.g. TP=4):
    python -m trainers_server.dp_worker.main --config config.json
    # mp.spawn is used automatically based on config.training.gpus

The world_size equals len(config.training.gpus) which must equal
tensor_parallel_size * pipeline_parallel_size.
"""

import argparse
import logging
import os
import socket
import warnings

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import uvicorn

from trainers_server.dp_worker.api.models import RLControllerConfig
from trainers_server.dp_worker.api.server import create_app
from trainers_server.dp_worker.api.controller import RLController, worker_loop

logger = logging.getLogger(__name__)


# ── Argument parsing ──────────────────────────────────────────────────


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="dp_worker: Megatron-based RL training server")
    p.add_argument("--config", required=True, help="Path to JSON config file")
    p.add_argument("--port", type=int, default=8000, help="HTTP port (rank 0 only)")
    p.add_argument("--host", default="0.0.0.0", help="HTTP host (rank 0 only)")
    return p.parse_args(argv)


# ── Per-rank entry point ──────────────────────────────────────────────


def _run_rank(
    rank: int,
    world_size: int,
    config: RLControllerConfig,
    dist_port: int,
    http_host: str,
    http_port: int,
) -> None:
    """Initialise distributed context and run either the HTTP server (rank 0)
    or the collective-op worker loop (ranks 1+)."""

    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s %(levelname)s [rank{rank}] %(name)s: %(message)s",
    )

    # ── Distributed init ──────────────────────────────────────────────
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(dist_port)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    # Map logical rank → physical GPU from config.training.gpus list.
    physical_gpu = config.training.gpus[rank]
    torch.cuda.set_device(physical_gpu)

    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
    )
    logger.info("Rank %d initialised on GPU %d", rank, physical_gpu)

    # ── Model + optimizer ─────────────────────────────────────────────
    # RLController.__init__ calls provide_distributed_model which is collective.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        controller = RLController(config)

    # ── Rank 0: HTTP server; others: worker loop ───────────────────────
    if rank == 0:
        app = create_app(controller=controller)
        logger.info("Starting HTTP server on %s:%d", http_host, http_port)
        uvicorn.run(app, host=http_host, port=http_port, log_level="info")

        # Server has stopped — signal all workers to exit.
        op_t = torch.tensor([255], dtype=torch.int32)  # OP_EXIT
        if dist.is_available() and dist.is_initialized():
            dist.broadcast(op_t, src=0)
    else:
        worker_loop(controller)

    # ── Cleanup ───────────────────────────────────────────────────────
    from megatron.core import parallel_state
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


# ── Free-port helper ──────────────────────────────────────────────────


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return int(s.getsockname()[1])


# ── Main ─────────────────────────────────────────────────────────────


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    args = parse_args()
    with open(args.config) as f:
        config = RLControllerConfig.model_validate_json(f.read())

    world_size = len(config.training.gpus)
    expected = config.training.tensor_parallel_size * config.training.pipeline_parallel_size
    if world_size != expected:
        raise ValueError(
            f"len(training.gpus)={world_size} must equal "
            f"tensor_parallel_size({config.training.tensor_parallel_size}) × "
            f"pipeline_parallel_size({config.training.pipeline_parallel_size})={expected}"
        )

    logger.info(
        "Launching %d worker(s): TP=%d PP=%d",
        world_size,
        config.training.tensor_parallel_size,
        config.training.pipeline_parallel_size,
    )

    dist_port = _find_free_port()

    if world_size == 1:
        # Single-process shortcut — no mp.spawn overhead.
        _run_rank(
            rank=0,
            world_size=1,
            config=config,
            dist_port=dist_port,
            http_host=args.host,
            http_port=args.port,
        )
    else:
        mp.spawn(
            _run_rank,
            args=(world_size, config, dist_port, args.host, args.port),
            nprocs=world_size,
            join=True,
        )


if __name__ == "__main__":
    main()
