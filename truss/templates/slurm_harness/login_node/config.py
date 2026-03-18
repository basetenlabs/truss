"""
SLURM Login/Controller Node Configuration

This is a CPU-only node that runs slurmctld (the SLURM controller daemon).
It stays alive to accept job submissions from worker nodes.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from shared.docker_auth import build_docker_auth
from truss.base import truss_config
from truss_train import definitions

config_path = Path(__file__).parent / "runtime_config.json"
runtime_config = json.loads(config_path.read_text()) if config_path.exists() else {}

GPUS_PER_NODE = runtime_config.get("gpus_per_node", 8)
PARTITION = runtime_config.get("partition", None)

BASE_IMAGE = runtime_config.get(
    "base_image", "pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime"
)
DOCKER_AUTH = build_docker_auth(
    BASE_IMAGE,
    runtime_config.get("docker_auth_method"),
    runtime_config.get("docker_auth_secret"),
)

training_runtime = definitions.Runtime(
    start_commands=[
        "apt-get update -qq && apt-get install -y -qq git > /dev/null 2>&1",
        "pip install --quiet 'truss @ git+https://github.com/basetenlabs/truss.git@rcano/slurm-cli'",
        "bash login_node/setup_login.sh",
    ],
    environment_variables={
        "GPUS_PER_NODE": str(GPUS_PER_NODE),
        "BASETEN_API_KEY": definitions.SecretReference(name="baseten_api_key"),
    },
    cache_config=definitions.CacheConfig(enabled=True, require_cache_affinity=False),
)

if PARTITION:
    training_compute = definitions.Compute(
        accelerator=truss_config.AcceleratorSpec(
            accelerator=truss_config.Accelerator(PARTITION), count=GPUS_PER_NODE
        )
    )
else:
    training_compute = definitions.Compute(cpu_count=4, memory="16Gi")

training_job = definitions.TrainingJob(
    image=definitions.Image(base_image=BASE_IMAGE, docker_auth=DOCKER_AUTH),
    compute=training_compute,
    runtime=training_runtime,
    interactive_session=definitions.InteractiveSession(
        trigger=definitions.InteractiveSessionTrigger.ON_STARTUP
    ),
    name=runtime_config.get("job_name"),
)

training_project = definitions.TrainingProject(
    name=runtime_config.get("project_name", "slurm-harness"), job=training_job
)
