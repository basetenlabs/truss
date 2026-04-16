"""CLI entry point for the dp_worker server.

Single-rank (existing behaviour)
---------------------------------
    python -m trainers_server.dp_worker.main --config cfg.json [--port 8000]

Multi-rank via torchrun
-----------------------
    torchrun --nproc_per_node=4 -m trainers_server.dp_worker.main --config cfg.json

When ``WORLD_SIZE > 1``:
  * Rank 0 initialises ``torch.distributed``, builds the ``RLController``, and
    starts the FastAPI/uvicorn server (same as the single-rank path).
  * Ranks 1…N-1 initialise ``torch.distributed``, build their own
    ``RLController`` instance, then enter ``controller.worker_loop()`` — a
    blocking loop that waits for broadcast op-codes from rank 0 and
    participates in the FSDP collective operations.
"""
from __future__ import annotations

import argparse
import logging

import uvicorn

from trainers_server.dp_worker.distributed import (
    destroy_process_group,
    init_process_group,
    is_rank_zero,
)
from trainers_server.dp_worker.api.models import RLControllerConfig
from trainers_server.dp_worker.api.server import create_app
from trainers_server.dp_worker.api.controller import RLController

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="dp_worker: RL training worker server")
    parser.add_argument("--config", required=True, help="Path to JSON config file")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    return parser.parse_args(argv)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()

    with open(args.config) as f:
        config_json = f.read()
    config = RLControllerConfig.model_validate_json(config_json)
    logger.info("loaded config: model_id=%s", config.model_id)

    # Initialize torch.distributed when running under torchrun (WORLD_SIZE > 1).
    # No-ops in single-rank mode.
    init_process_group()

    controller = RLController(config)

    if is_rank_zero():
        # Rank 0 runs the HTTP server; the server is responsible for broadcasting
        # op-codes to worker ranks before each collective operation.
        logger.info("Rank 0: starting FastAPI server on %s:%d", args.host, args.port)
        app = create_app(controller=controller)
        try:
            uvicorn.run(app, host=args.host, port=args.port)
        finally:
            controller.close()
            destroy_process_group()
    else:
        # Worker ranks block here, executing FSDP collectives on demand.
        try:
            controller.worker_loop()
        finally:
            controller.close()
            destroy_process_group()


if __name__ == "__main__":
    main()
