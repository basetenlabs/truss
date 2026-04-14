"""Deploy a training worker and return a TrainingClient connected directly to it."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Optional

import httpx

from trainers.training_client import TrainingClient
from truss.base import truss_config
from truss_train import definitions
from truss_train.public_api import push


_DEFAULT_WORKSPACE_ROOT = Path(__file__).resolve().parent.parent / "server"


def create_training_client(
    base_model: str,
    worker_url: str,
    *,
    api_key: str | None = None,
    gpu_count: int = 1,
    accelerator: str = "H100",
    training_gpus: Optional[list[int]] = None,
    inference_gpus: Optional[list[int]] = None,
    max_seq_len: int = 4096,
    worker_port: int = 8001,
    namespace: str = "default",
    remote: str = "baseten",
    workspace_root: Optional[Path] = None,
    deploy: bool = True,
    timeout: float = 600.0,
) -> TrainingClient:
    """Deploy a training worker and return a TrainingClient connected to it.

    Args:
        base_model: HuggingFace model ID (e.g. "Qwen/Qwen3-8B").
        worker_url: URL of the dp_worker to connect to. If deploy=True, this
            is computed automatically from the job ID and namespace.
        api_key: API key for authentication.
        gpu_count: Total number of GPUs to request.
        accelerator: GPU type (H100, H200, B200).
        training_gpus: GPU indices for training. Defaults to [0].
        inference_gpus: GPU indices for inference. Defaults to [0].
        max_seq_len: Maximum sequence length for training.
        worker_port: Port the dp_worker will listen on.
        namespace: K8s namespace where the training job runs.
        remote: Baseten remote name from .trussrc.
        workspace_root: Path to thinker workspace root.
        deploy: If True, deploy a training job. If False, just connect to worker_url.
        timeout: HTTP timeout for training operations.
    """
    if not deploy:
        return TrainingClient(worker_url, api_key=api_key, timeout=timeout)

    suffix = uuid.uuid4().hex[:7]
    project_name = f"trainer-{base_model.replace('/', '-')}-{suffix}"
    print(f"Project: {project_name}")

    if training_gpus is None:
        training_gpus = [0]
    if inference_gpus is None:
        inference_gpus = [0]

    rl_config = {
        "model_id": base_model,
        "training": {
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "max_length": max_seq_len,
            "gpus": training_gpus,
        },
        "inference": {
            "tensor_parallel_size": 1,
            "gpus": inference_gpus,
            "gpu_memory_utilization": 0.9,
        },
    }

    ws_root = workspace_root or _DEFAULT_WORKSPACE_ROOT
    if not ws_root.exists():
        raise FileNotFoundError(
            f"Workspace root not found: {ws_root}. "
            "Pass workspace_root= pointing to the thinker repo."
        )

    rl_config_path = ws_root / "rl_config.json"
    rl_config_path.write_text(json.dumps(rl_config))

    accel_enum = getattr(
        truss_config.Accelerator,
        accelerator.upper(),
        truss_config.Accelerator.H100,
    )

    project = definitions.TrainingProject(
        name=project_name,
        job=definitions.TrainingJob(
            compute=definitions.Compute(
                accelerator=truss_config.AcceleratorSpec(
                    accelerator=accel_enum,
                    count=gpu_count,
                ),
            ),
            runtime=definitions.Runtime(
                start_commands=[
                    "apt-get update && apt-get install -y python3-dev curl",
                    "curl -LsSf https://astral.sh/uv/install.sh | sh",
                    ". $HOME/.local/bin/env && uv sync --extra worker",
                    f".venv/bin/python -m trainers_server.dp_worker.main --config $RL_CONFIG_PATH --port {worker_port}",
                ],
                environment_variables={
                    "RL_CONFIG_PATH": "rl_config.json",
                    "BASETEN_API_KEY": definitions.SecretReference(name="baseten_api_key"),
                },
            ),
            image=definitions.Image(
                base_image="nvcr.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04",
            ),
            workspace=definitions.Workspace(
                workspace_root=str(ws_root),
                exclude_dirs=[
                    str(ws_root / ".venv"),
                    str(ws_root / ".git"),
                ],
            ),
        ),
    )

    try:
        result = push(project, remote=remote, source_dir=ws_root)
        job_id = result["id"]
        print(f"Training Job ID: {job_id}")
    finally:
        rl_config_path.unlink(missing_ok=True)

    # Build the deterministic pod DNS for the worker.
    worker_host = (
        f"baseten-training-job-{job_id}-multinode-0"
        f".baseten-training-job-{job_id}-multinode"
        f".{namespace}.svc.cluster.local"
    )
    resolved_url = f"http://{worker_host}:{worker_port}"
    print(f"Worker URL: {resolved_url}")

    return TrainingClient(resolved_url, api_key=api_key, timeout=timeout)
