# Import necessary classes from the Baseten Training SDK
from truss_train import definitions

# Project name - change this to identify your training project
PROJECT_NAME = "MNIST Training Example"

# Base image with PyTorch pre-installed
BASE_IMAGE = "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime"

# Runtime configuration
training_runtime = definitions.Runtime(
    start_commands=[
        "/bin/sh -c 'chmod +x ./run.sh && ./run.sh'",
    ],
    environment_variables={
        # Add your secrets here (configure in Baseten workspace first):
        # "HF_TOKEN": definitions.SecretReference(name="hf_access_token"),
        # "WANDB_API_KEY": definitions.SecretReference(name="wandb_api_key"),
    },
    # Enable checkpointing to save model weights to Baseten
    checkpointing_config=definitions.CheckpointingConfig(
        enabled=True,
    ),
    # Enable caching for datasets and other reusable files
    cache_config=definitions.CacheConfig(
        enabled=True,
    ),
)

# Compute configuration - CPU only for this example
# For GPU training, uncomment and modify the accelerator section
training_compute = definitions.Compute(
    cpu_count=4,
    memory="16Gi",
    # accelerator=truss_config.AcceleratorSpec(
    #     accelerator=truss_config.Accelerator.H100,
    #     count=1,
    # ),
)

# Combine into a training job
training_job = definitions.TrainingJob(
    image=definitions.Image(base_image=BASE_IMAGE),
    compute=training_compute,
    runtime=training_runtime,
)

# Define the training project
training_project = definitions.TrainingProject(name=PROJECT_NAME, job=training_job)
