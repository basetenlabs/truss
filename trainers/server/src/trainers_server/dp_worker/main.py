"""CLI entry point for the dp_worker server."""

import argparse
import logging

import uvicorn

from trainers_server.dp_worker.api.models import RLControllerConfig
from trainers_server.dp_worker.api.server import create_app

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="dp_worker: RL training worker server")
    parser.add_argument("--config", required=True, help="Path to JSON config file")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    return parser.parse_args(argv)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()

    with open(args.config) as f:
        config_json = f.read()

    config = RLControllerConfig.model_validate_json(config_json)
    logger.info("loaded config: model_id=%s", config.model_id)

    app = create_app(config)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
