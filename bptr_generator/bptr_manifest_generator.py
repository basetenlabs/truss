#!/usr/bin/env python3
import argparse
import json
import logging
import sys
from pathlib import Path

import yaml
from truss_transfer import PyModelRepo, create_basetenpointer_from_models

logging.basicConfig(level=logging.INFO, format="%(message)s")


def load_truss_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    if "model_cache" not in config:
        raise ValueError("No 'model_cache' found in config.yaml")
    return config


def load_model_cache_config(config_path: Path) -> list[dict]:
    """Load model_cache config directly from JSON file"""
    if not config_path.exists():
        raise FileNotFoundError(f"Model cache config file not found: {config_path}")
    with open(config_path, "r") as f:
        model_cache = json.load(f)
    if not isinstance(model_cache, list):
        raise ValueError("Model cache config must be a list of model configurations")
    return model_cache


def generate_bptr_manifest_from_model_cache(
    model_cache: list[dict],
    output_path: Path,
    model_path: str = "/cache/model/model_cache",
):
    """Generate bptr-manifest directly from model_cache configuration"""
    if not any(model.get("use_volume", False) for model in model_cache):
        raise ValueError("No v2 models found (use_volume=True)")
    if not all(
        model.get("volume_folder")
        for model in model_cache
        if model.get("use_volume", False)
    ):
        raise ValueError("All v2 models must have volume_folder")

    py_models = [
        PyModelRepo(
            repo_id=model["repo_id"],
            revision=model.get("revision"),
            runtime_secret_name=model.get("runtime_secret_name", "hf_access_token"),
            allow_patterns=model.get("allow_patterns"),
            ignore_patterns=model.get("ignore_patterns"),
            volume_folder=model.get("volume_folder"),
            kind=model.get("kind", "hf"),
        )
        for model in model_cache
        if model.get("use_volume", False)
    ]

    basetenpointer_json = create_basetenpointer_from_models(
        models=py_models, model_path=model_path
    )
    bptr_py = json.loads(basetenpointer_json)["pointers"]
    logging.info(f"created ({len(bptr_py)}) Basetenpointer")
    logging.info(f"pointers json: {basetenpointer_json}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(basetenpointer_json)


def generate_bptr_manifest(
    config: dict, output_path: Path, model_path: str = "/cache/model/model_cache"
):
    """Legacy function for backward compatibility"""
    model_cache = config.get("model_cache", [])
    generate_bptr_manifest_from_model_cache(model_cache, output_path, model_path)


def main():
    parser = argparse.ArgumentParser(
        description="Generate bptr-manifest from truss config.yaml or model_cache config"
    )
    parser.add_argument(
        "--config", "-c", type=Path, help="Path to truss config.yaml file"
    )
    parser.add_argument(
        "--model-cache", "-m", type=Path, help="Path to model_cache JSON config file"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Output path for bptr-manifest file",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="/cache/model/model_cache",
        help="Base path for model files in manifest",
    )
    args = parser.parse_args()

    if not args.config and not args.model_cache:
        parser.error("Either --config or --model-cache must be specified")
    if args.config and args.model_cache:
        parser.error("Cannot specify both --config and --model-cache")

    try:
        if args.model_cache:
            model_cache = load_model_cache_config(args.model_cache)
            generate_bptr_manifest_from_model_cache(
                model_cache, args.output, args.model_path
            )
        else:
            config = load_truss_config(args.config)
            generate_bptr_manifest_from_model_cache(
                config.get("model_cache", []), args.output, args.model_path
            )
    except Exception as e:
        logging.error(f"Error generating bptr-manifest: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
