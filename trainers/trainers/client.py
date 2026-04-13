"""Deploy a training worker and return a TrainingClient connected to TRM."""

from __future__ import annotations

import configparser
import json
import uuid
from pathlib import Path
from typing import Optional

import httpx

from trainers.training_client import TrainingClient
from trainers.queue_client import QueueClient
from truss.base import truss_config
from truss_train import definitions
from truss_train.public_api import push


_DEFAULT_WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent.parent / "thinker-container" / "thinker"


def _get_baseten_api_key(remote: str) -> str:
    config = configparser.ConfigParser()
    config.read(Path.home() / ".trussrc")
    return config[remote]["api_key"]


def _register_backend(
    trm_url: str,
    api_key: str,
    backend_id: str,
    client_id: str,
    backend_type: str,
    base_url: str,
) -> None:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Api-Key {api_key}"
    resp = httpx.post(
        f"{trm_url}/register_backend",
        json={
            "id": backend_id,
            "client_id": client_id,
            "backend_type": backend_type,
            "base_url": base_url,
        },
        headers=headers,
        timeout=30,
    )
    resp.raise_for_status()


def create_training_client(
    base_model: str,
    trm_url: str,
    trm_api_key: str = "",
    *,
    namespace: str,
    gpu_count: int = 1,
    accelerator: str = "H100",
    training_gpus: Optional[list[int]] = None,
    inference_gpus: Optional[list[int]] = None,
    max_seq_len: int = 4096,
    worker_port: int = 8001,
    remote: str = "baseten",
    workspace_root: Optional[Path] = None,
) -> TrainingClient:
    """Deploy a training worker and return a TrainingClient connected to TRM.

    Args:
        base_model: HuggingFace model ID (e.g. "Qwen/Qwen3-8B").
        trm_url: URL of the training-request-manager service.
        trm_api_key: API key for TRM authentication.
        namespace: K8s namespace where the training job runs (e.g. "org-{org_id}").
        gpu_count: Total number of GPUs to request.
        accelerator: GPU type (H100, H200, B200).
        training_gpus: GPU indices for training. Defaults to [0].
        inference_gpus: GPU indices for inference. Defaults to [0].
        max_seq_len: Maximum sequence length for training.
        worker_port: Port the dp_worker will listen on.
        remote: Baseten remote name from .trussrc.
        workspace_root: Path to thinker workspace root. Auto-detected if None.
    """
    suffix = uuid.uuid4().hex[:7]
    project_name = f"trainer-{base_model.replace('/', '-')}-{suffix}"
    client_id = suffix
    print(f"Project: {project_name} (client_id: {client_id})")

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
                    f".venv/bin/python -m thinker.dp_worker.main --config $RL_CONFIG_PATH --port {worker_port}",
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

    # Register with the deterministic StatefulSet pod DNS.
    # Training pods are: baseten-training-job-{job_id}-multinode-0
    # in the given namespace.
    worker_host = (
        f"baseten-training-job-{job_id}-multinode-0"
        f".baseten-training-job-{job_id}-multinode"
        f".{namespace}.svc.cluster.local"
    )
    _register_backend(
        trm_url=trm_url,
        api_key=trm_api_key,
        backend_id=job_id,
        client_id=client_id,
        backend_type="training",
        base_url=f"http://{worker_host}:{worker_port}",
    )
    print(f"Registered training backend {job_id} at {worker_host}:{worker_port}")

    if trm_api_key:
        http_client = httpx.Client(
            headers={"Authorization": f"Api-Key {trm_api_key}"},
            timeout=30,
        )
        queue_client = QueueClient(trm_url, client=http_client)
        return TrainingClient(trm_url, client=queue_client)
    return TrainingClient(trm_url)
