"""CLI entry point for the dp_worker server."""

import argparse
import logging
import os

import uvicorn

from trainers_server.dp_worker.api.models import (
    InferenceServerConfig,
    RLControllerConfig,
    TrainingServerConfig,
)
from trainers_server.dp_worker.api.server import create_app

logger = logging.getLogger(__name__)


def _config_from_env() -> RLControllerConfig | None:
    """Build RLControllerConfig from environment variables.

    Returns None if BT_MODEL_ID is not set (fall back to --config JSON).

    Environment variables:
        BT_MODEL_ID             HuggingFace model ID (required to activate env-var mode)
        BT_LORA_RANK            LoRA rank; 0 = full fine-tuning (default: 0)
        BT_MAX_LENGTH           Max token sequence length for training (default: 2048)
        BT_TRAINING_GPUS        Comma-separated GPU IDs for training, e.g. "0,1" (default: "0")
        BT_TENSOR_PARALLEL_SIZE Tensor parallel size for training (default: 1)
    """
    model_id = os.environ.get("BT_MODEL_ID")
    if model_id is None:
        return None

    training_gpus_raw = os.environ.get("BT_TRAINING_GPUS", "0")
    training_gpus = [int(g.strip()) for g in training_gpus_raw.split(",") if g.strip()]

    return RLControllerConfig(
        model_id=model_id,
        lora_rank=int(os.environ.get("BT_LORA_RANK", "0")),
        training=TrainingServerConfig(
            max_length=int(os.environ.get("BT_MAX_LENGTH", "2048")),
            gpus=training_gpus,
            tensor_parallel_size=int(os.environ.get("BT_TENSOR_PARALLEL_SIZE", "1")),
        ),
        inference=InferenceServerConfig(),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="dp_worker: RL training worker server")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to JSON config file. Not required if BT_MODEL_ID env var is set.",
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to listen on (default: 8000)"
    )
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)"
    )
    return parser.parse_args(argv)


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = parse_args()

    config = _config_from_env()
    if config is not None:
        logger.info("loaded config from environment: model_id=%s", config.model_id)
    elif args.config is not None:
        with open(args.config) as f:
            config_json = f.read()
        config = RLControllerConfig.model_validate_json(config_json)
        logger.info(
            "loaded config from file %s: model_id=%s", args.config, config.model_id
        )
    else:
        raise SystemExit(
            "error: must supply either BT_MODEL_ID env var or --config <path>"
        )

    app = create_app(config)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
