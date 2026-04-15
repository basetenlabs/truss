"""Training job config template for deploying dp_worker.

This can be used directly with `truss train push config.py` or as a reference
for the programmatic API in trainers.create_training_client().

Placeholders (in braces) are replaced at runtime when using create_training_client().
"""

from truss_train import definitions
from truss.base import truss_config

runtime = definitions.Runtime(
    start_commands=[
        "apt-get update && apt-get install -y python3-dev curl git",
        "curl -LsSf https://astral.sh/uv/install.sh | sh",
        ". $HOME/.local/bin/env && uv sync --extra worker",
        ".venv/bin/python -m trainers_server.dp_worker.main --config $RL_CONFIG_PATH --port {WORKER_PORT}",
    ],
    environment_variables={
        "RL_CONFIG_PATH": "rl_config.json",
        "BASETEN_API_KEY": definitions.SecretReference(name="baseten_api_key"),
    },
)

training_job = definitions.TrainingJob(
    compute=definitions.Compute(
        accelerator=truss_config.AcceleratorSpec(
            accelerator=truss_config.Accelerator.H200,
            count=2,
        ),
    ),
    runtime=runtime,
    image=definitions.Image(
        base_image="nvcr.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04",
    ),
    workspace=definitions.Workspace(
        workspace_root="../../../../../",
        exclude_dirs=["../../../../../.venv", "../../../../../.git"],
    ),
)

first_project = definitions.TrainingProject(
    name="{PROJECT_NAME}",
    job=training_job,
)
