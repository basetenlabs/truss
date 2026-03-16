"""
SLURM Worker Node Configuration

Worker nodes run slurmd and register with the controller.
Worker node 0 submits the sbatch job once all workers are ready.
"""

import json
from pathlib import Path

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

BASE_IMAGE = runtime_config.get("base_image", "pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime")

training_runtime = definitions.Runtime(
    start_commands=[
        "bash worker_node/setup_worker.sh",
    ],
    environment_variables={
        "EXPECTED_WORKERS": str(NODE_COUNT),
        "GPUS_PER_NODE": str(GPUS_PER_NODE),
        "SBATCH_SCRIPT": SBATCH_SCRIPT,
    },
    cache_config=definitions.CacheConfig(
        enabled=True,
    ),
    checkpointing_config=definitions.CheckpointingConfig(
        enabled=True,
    ),
)

training_compute = definitions.Compute(
    node_count=NODE_COUNT,
    accelerator=truss_config.AcceleratorSpec(
        accelerator=accelerator_type,
        count=GPUS_PER_NODE,
    ),
)

training_job = definitions.TrainingJob(
    image=definitions.Image(base_image=BASE_IMAGE),
    compute=training_compute,
    runtime=training_runtime,
    interactive_session=definitions.InteractiveSession(
        trigger=definitions.InteractiveSessionTrigger.ON_STARTUP,
    ),
    name=runtime_config.get("job_name"),
)

training_project = definitions.TrainingProject(
    name=runtime_config.get("project_name", "slurm-harness"),
    job=training_job,
)
