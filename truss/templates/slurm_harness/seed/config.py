"""
Seed Job Configuration

Pushes a minimal 1x H200 job to establish the project cache on H200
infrastructure. This is a hack to ensure cache affinity for subsequent
GPU worker jobs. The seed job exits immediately after starting.
"""

import json
from pathlib import Path

from truss.base import truss_config
from truss_train import definitions

config_path = Path(__file__).parent / "runtime_config.json"
if config_path.exists():
    runtime_config = json.loads(config_path.read_text())
else:
    runtime_config = {}

BASE_IMAGE = runtime_config.get("base_image", "pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime")

training_runtime = definitions.Runtime(
    start_commands=[
        "echo 'SEED_READY' && sleep 30",
    ],
    cache_config=definitions.CacheConfig(
        enabled=True,
    ),
)

training_compute = definitions.Compute(
    accelerator=truss_config.AcceleratorSpec(
        accelerator=truss_config.Accelerator.H200,
        count=1,
    ),
)

training_job = definitions.TrainingJob(
    image=definitions.Image(base_image=BASE_IMAGE),
    compute=training_compute,
    runtime=training_runtime,
)

training_project = definitions.TrainingProject(
    name=runtime_config.get("project_name", "slurm-harness"),
    job=training_job,
)
