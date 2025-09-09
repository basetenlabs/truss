# Import necessary classes from the Baseten Training SDK
from truss_train import definitions
from truss.base import truss_config

PROJECT_NAME = "My-Baseten-Training-Project"
NUM_NODES = 1
NUM_GPUS_PER_NODE = 1

# 1. Define a base image for your training job. You can also use
# private images via AWS IAM or GCP Service Account authentication.
BASE_IMAGE = "pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime"

# 2. Define the Runtime Environment for the Training Job
# This includes start commands and environment variables.
# Secrets from the baseten workspace like API keys are referenced using
# `SecretReference`.
training_runtime = definitions.Runtime(
    start_commands=[  # Example: list of commands to run your training script
        "/bin/sh -c 'chmod +x ./run.sh && ./run.sh'"
    ],
    environment_variables={
        # "HF_TOKEN": definitions.SecretReference(name="hf_access_token"),
        "HELLO": "WORLD"
    },
    cache_config=definitions.CacheConfig(
        enabled=False  # Set to True to enable caching between runs
    ),
    checkpointing_config=definitions.CheckpointingConfig(
        enabled=False  # Set to True to enable saving checkpoints on Baseten
    ),
)

training_compute = definitions.Compute(
    node_count=NUM_NODES,
    accelerator=truss_config.AcceleratorSpec(
        accelerator=truss_config.Accelerator.H100, count=NUM_GPUS_PER_NODE
    ),
)

training_job = definitions.TrainingJob(
    image=definitions.Image(base_image=BASE_IMAGE),
    compute=training_compute,
    runtime=training_runtime,
)

training_project = definitions.TrainingProject(name=PROJECT_NAME, job=training_job)
