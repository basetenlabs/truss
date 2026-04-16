"""Tests for weight export (to_inference / save_state).

These tests verify that megatron-bridge's save_hf_pretrained correctly exports
Megatron checkpoint weights to HuggingFace format and that the output can be
loaded back with transformers. The vLLM reload step is mocked.

Run:
    cd trainers/server
    uv run --extra worker pytest tests/test_weight_sync.py -v -s
"""

import json
import os
import socket
import tempfile
import warnings
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from helpers import MODEL_PATH, skip_if_no_gpu

pytestmark = pytest.mark.gpu


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _init_dist(rank: int, world_size: int, dist_port: int, gpu: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(dist_port)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    torch.cuda.set_device(gpu)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


def _teardown() -> None:
    try:
        from megatron.core import parallel_state
        parallel_state.destroy_model_parallel()
    except Exception:
        pass
    dist.destroy_process_group()


def _start_vllm_patches(controller_class):
    """Patch all vLLM-related methods on RLController and return the patches."""
    patches = [
        patch.object(controller_class, "_launch_rollout"),
        patch.object(controller_class, "_wait_for_rollout"),
        patch.object(controller_class, "_kill_rollout"),
    ]
    for p in patches:
        p.start()
    return patches


def _stop_patches(patches):
    for p in reversed(patches):
        p.stop()


# ── Test: to_inference exports valid HF checkpoint ───────────────────────────


def _to_inference_rank(
    rank: int, world_size: int, dist_port: int, export_path: str, result_path: str
) -> None:
    """All ranks call _execute_to_inference(); rank 0 verifies the output."""
    gpus = [0]
    _init_dist(rank, world_size, dist_port, gpus[rank])

    import trainers_server.dp_worker.api.controller as ctrl_module
    from trainers_server.dp_worker.api.controller import RLController
    from trainers_server.dp_worker.api.models import (
        RLControllerConfig, TrainingServerConfig, InferenceServerConfig
    )

    config = RLControllerConfig(
        model_id=MODEL_PATH,
        training=TrainingServerConfig(
            tensor_parallel_size=1, pipeline_parallel_size=1, gpus=gpus, max_length=64
        ),
        inference=InferenceServerConfig(tensor_parallel_size=1, gpus=[0]),
    )

    # Keep vLLM patches active through both __init__ and _execute_to_inference.
    # _execute_to_inference calls _kill_rollout + _launch_rollout + _wait_for_rollout
    # on rank 0; we don't want a real vLLM subprocess for this test.
    patches = _start_vllm_patches(RLController)
    orig_sync_path = ctrl_module._WEIGHT_SYNC_PATH
    ctrl_module._WEIGHT_SYNC_PATH = export_path

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            controller = RLController(config)

        # Collective: all ranks call this.
        controller._execute_to_inference()
    finally:
        ctrl_module._WEIGHT_SYNC_PATH = orig_sync_path
        _stop_patches(patches)

    if rank == 0:
        export_dir = Path(export_path)
        files = {f.name for f in export_dir.iterdir()}

        config_json = json.loads((export_dir / "config.json").read_text())
        with open(result_path, "w") as f:
            json.dump({
                "has_config": "config.json" in files,
                "has_weights": any(
                    fn.endswith(".safetensors") or fn.endswith(".bin")
                    for fn in files
                ),
                "num_hidden_layers": config_json.get("num_hidden_layers", -1),
                "model_type": config_json.get("model_type", ""),
                "files": sorted(files),
            }, f)

    _teardown()


@skip_if_no_gpu(1)
def test_to_inference_exports_hf_checkpoint(tmp_path):
    """_execute_to_inference() should write a valid HF checkpoint to disk."""
    export_path = str(tmp_path / "hf_export")
    os.makedirs(export_path, exist_ok=True)

    dist_port = _find_free_port()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        result_path = f.name

    try:
        mp.spawn(
            _to_inference_rank,
            args=(1, dist_port, export_path, result_path),
            nprocs=1,
            join=True,
        )
        with open(result_path) as f:
            results = json.load(f)
    finally:
        Path(result_path).unlink(missing_ok=True)

    assert results["has_config"], f"config.json missing from export. Files: {results['files']}"
    assert results["has_weights"], f"No weight files in export. Files: {results['files']}"
    assert results["num_hidden_layers"] > 0, "config.json missing num_hidden_layers"


@skip_if_no_gpu(1)
def test_to_inference_checkpoint_loads_with_transformers(tmp_path):
    """Exported HF checkpoint can be loaded by transformers (config check)."""
    export_path = str(tmp_path / "hf_export_load")
    os.makedirs(export_path, exist_ok=True)

    dist_port = _find_free_port()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        result_path = f.name

    try:
        mp.spawn(
            _to_inference_rank,
            args=(1, dist_port, export_path, result_path),
            nprocs=1,
            join=True,
        )
    finally:
        Path(result_path).unlink(missing_ok=True)

    # Load config in the parent process (no GPU or dist needed).
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(export_path)
    assert cfg is not None
    assert hasattr(cfg, "num_hidden_layers")
    assert cfg.num_hidden_layers > 0


# ── Test: save_state writes HF checkpoint + trainer_state.pt ─────────────────


def _save_state_rank(
    rank: int, world_size: int, dist_port: int, save_path: str, result_path: str
) -> None:
    gpus = [0]
    _init_dist(rank, world_size, dist_port, gpus[rank])

    from trainers_server.dp_worker.api.controller import RLController, OP_EXIT, worker_loop
    from trainers_server.dp_worker.api.models import (
        RLControllerConfig, TrainingServerConfig, InferenceServerConfig
    )
    from trainers_server.shared.models import (
        AdamParams, Datum, ForwardBackwardDetails, ModelInput,
        OptimStepDetails, TensorData,
    )

    config = RLControllerConfig(
        model_id=MODEL_PATH,
        training=TrainingServerConfig(
            tensor_parallel_size=1, pipeline_parallel_size=1, gpus=gpus, max_length=64
        ),
        inference=InferenceServerConfig(tensor_parallel_size=1, gpus=[0]),
    )

    patches = _start_vllm_patches(RLController)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            controller = RLController(config)

        data = [Datum(
            model_input=ModelInput.from_ints(list(range(1, 9))),
            loss_fn_inputs={"reward": TensorData(data=[1.0], dtype="float32", shape=[1])},
        )]
        details = ForwardBackwardDetails(data=data)

        if rank == 0:
            # All three calls go through the broadcast layer (op-code → workers).
            # With world_size=1 those broadcasts are trivial no-ops.
            controller.forward_backward(details)
            controller.optim_step(OptimStepDetails(adam_params=AdamParams()))
            os.makedirs(save_path, exist_ok=True)
            controller.save_state(save_path)  # broadcasts OP_SAVE_STATE

            # Signal exit after all collective work is done.
            op_t = torch.tensor([255], dtype=torch.int32)  # OP_EXIT
            dist.broadcast(op_t, src=0)
        else:
            # Workers handle OP_FORWARD_BACKWARD, OP_OPTIM_STEP, OP_SAVE_STATE,
            # then exit on OP_EXIT.
            worker_loop(controller)
    finally:
        _stop_patches(patches)

    if rank == 0:
        save_dir = Path(save_path)
        files = {f.name for f in save_dir.iterdir()}
        trainer_state = torch.load(save_dir / "trainer_state.pt", weights_only=False)

        with open(result_path, "w") as f:
            json.dump({
                "has_config": "config.json" in files,
                "has_trainer_state": "trainer_state.pt" in files,
                "has_weights": any(
                    fn.endswith(".safetensors") or fn.endswith(".bin")
                    for fn in files
                ),
                "step": trainer_state["step"],
                "mode": trainer_state["mode"],
                "files": sorted(files),
            }, f)

    _teardown()


@skip_if_no_gpu(1)
def test_save_state_writes_checkpoint_and_trainer_state(tmp_path):
    save_path = str(tmp_path / "checkpoint_step1")
    dist_port = _find_free_port()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        result_path = f.name

    try:
        mp.spawn(
            _save_state_rank,
            args=(1, dist_port, save_path, result_path),
            nprocs=1,
            join=True,
        )
        with open(result_path) as f:
            results = json.load(f)
    finally:
        Path(result_path).unlink(missing_ok=True)

    assert results["has_config"], f"config.json missing. Files: {results['files']}"
    assert results["has_trainer_state"], f"trainer_state.pt missing. Files: {results['files']}"
    assert results["has_weights"], f"No weight files. Files: {results['files']}"
    assert results["step"] == 1, f"Expected step=1, got {results['step']}"
    assert results["mode"] == "training"


# ── Test: TP=2 weight sync produces valid checkpoint ─────────────────────────


def _tp2_weight_sync_rank(
    rank: int, world_size: int, dist_port: int, export_path: str, result_path: str
) -> None:
    gpus = [0, 1]
    _init_dist(rank, world_size, dist_port, gpus[rank])

    import trainers_server.dp_worker.api.controller as ctrl_module
    from trainers_server.dp_worker.api.controller import RLController
    from trainers_server.dp_worker.api.models import (
        RLControllerConfig, TrainingServerConfig, InferenceServerConfig
    )

    config = RLControllerConfig(
        model_id=MODEL_PATH,
        training=TrainingServerConfig(
            tensor_parallel_size=2, pipeline_parallel_size=1, gpus=gpus, max_length=64
        ),
        inference=InferenceServerConfig(tensor_parallel_size=1, gpus=[2]),
    )

    patches = _start_vllm_patches(RLController)
    orig_sync_path = ctrl_module._WEIGHT_SYNC_PATH
    ctrl_module._WEIGHT_SYNC_PATH = export_path
    os.makedirs(export_path, exist_ok=True)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            controller = RLController(config)

        controller._execute_to_inference()
    finally:
        ctrl_module._WEIGHT_SYNC_PATH = orig_sync_path
        _stop_patches(patches)

    if rank == 0:
        export_dir = Path(export_path)
        files = {f.name for f in export_dir.iterdir()}
        config_data = json.loads((export_dir / "config.json").read_text())

        with open(result_path, "w") as f:
            json.dump({
                "has_config": "config.json" in files,
                "has_weights": any(
                    fn.endswith(".safetensors") or fn.endswith(".bin")
                    for fn in files
                ),
                "num_hidden_layers": config_data.get("num_hidden_layers", -1),
            }, f)

    _teardown()


@skip_if_no_gpu(2)
def test_tp2_weight_sync(tmp_path):
    """TP=2 save_hf_pretrained merges tensor-parallel shards into one HF checkpoint."""
    export_path = str(tmp_path / "tp2_export")
    dist_port = _find_free_port()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        result_path = f.name

    try:
        mp.spawn(
            _tp2_weight_sync_rank,
            args=(2, dist_port, export_path, result_path),
            nprocs=2,
            join=True,
        )
        with open(result_path) as f:
            results = json.load(f)
    finally:
        Path(result_path).unlink(missing_ok=True)

    assert results["has_config"], "TP=2 export missing config.json"
    assert results["has_weights"], "TP=2 export missing weight files"
    assert results["num_hidden_layers"] > 0, "TP=2 export config.json has bad num_hidden_layers"
