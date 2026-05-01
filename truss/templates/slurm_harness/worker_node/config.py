"""
SLURM Worker Node Configuration

Worker nodes run slurmd and register with the controller.
Worker node 0 submits the sbatch job once all workers are ready.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from shared.docker_auth import build_docker_auth
from truss.base import truss_config
from truss_train import definitions

# Read runtime configuration (written by push helper before push)
config_path = Path(__file__).parent / "runtime_config.json"
runtime_config = json.loads(config_path.read_text()) if config_path.exists() else {}

NODE_COUNT = runtime_config.get("node_count", 1)
GPUS_PER_NODE = runtime_config.get("gpus_per_node", 8)
PARTITION = runtime_config.get("partition", "H200")
SBATCH_SCRIPT = runtime_config.get("sbatch_script", "")

accelerator_type = truss_config.Accelerator(PARTITION)

BASE_IMAGE = runtime_config.get(
    "base_image", "pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime"
)
DOCKER_AUTH = build_docker_auth(
    BASE_IMAGE,
    runtime_config.get("docker_auth_method"),
    runtime_config.get("docker_auth_secret"),
)

training_runtime = definitions.Runtime(
    start_commands=["bash worker_node/setup_worker.sh"],
    environment_variables={
        "EXPECTED_WORKERS": str(NODE_COUNT),
        "GPUS_PER_NODE": str(GPUS_PER_NODE),
        "SBATCH_SCRIPT": SBATCH_SCRIPT,
    },
    cache_config=definitions.CacheConfig(enabled=True),
    checkpointing_config=definitions.CheckpointingConfig(
        enabled=runtime_config.get("checkpointing", True),
        checkpoint_path=runtime_config.get("checkpoint_path"),
        volume_size_gib=runtime_config.get("checkpoint_volume_size"),
    ),
)

training_compute = definitions.Compute(
    node_count=NODE_COUNT,
    accelerator=truss_config.AcceleratorSpec(
        accelerator=accelerator_type, count=GPUS_PER_NODE
    ),
)

_ISESSION_TRIGGERS = {
    "on_startup": definitions.InteractiveSessionTrigger.ON_STARTUP,
    "on_failure": definitions.InteractiveSessionTrigger.ON_FAILURE,
    "on_demand": definitions.InteractiveSessionTrigger.ON_DEMAND,
}
_ISESSION_PROVIDERS = {
    "vs_code": definitions.InteractiveSessionProvider.VS_CODE,
    "cursor": definitions.InteractiveSessionProvider.CURSOR,
}
_ISESSION_AUTH_PROVIDERS = {
    "microsoft": definitions.InteractiveSessionAuthProvider.MICROSOFT,
    "github": definitions.InteractiveSessionAuthProvider.GITHUB,
}
_isession_trigger = runtime_config.get("interactive_session", "on_demand")
_isession_kwargs = {}
if _isession_trigger in _ISESSION_TRIGGERS:
    _session_args = {"trigger": _ISESSION_TRIGGERS[_isession_trigger]}
    _sp = runtime_config.get("session_provider")
    if _sp and _sp in _ISESSION_PROVIDERS:
        _session_args["session_provider"] = _ISESSION_PROVIDERS[_sp]
    _ap = runtime_config.get("auth_provider")
    if _ap and _ap in _ISESSION_AUTH_PROVIDERS:
        _session_args["auth_provider"] = _ISESSION_AUTH_PROVIDERS[_ap]
    _isession_kwargs["interactive_session"] = definitions.InteractiveSession(
        **_session_args
    )

training_job = definitions.TrainingJob(
    image=definitions.Image(base_image=BASE_IMAGE, docker_auth=DOCKER_AUTH),
    compute=training_compute,
    runtime=training_runtime,
    **_isession_kwargs,
    name=runtime_config.get("job_name"),
)

training_project = definitions.TrainingProject(
    name=runtime_config.get("project_name", "slurm-harness"), job=training_job
)
