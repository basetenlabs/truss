from truss.base import truss_config
from truss_train import definitions

# PyTorch devel image: has Python, /workspace, CUDA toolkit + cuDNN for building.
# Pre-installed torch is ignored - our uv venv installs everything fresh.
BASE_IMAGE = "pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel"
ACCELERATOR = truss_config.Accelerator.H100
NGPU = 8
NODE_COUNT = 2

training_project = definitions.TrainingProject(
    name="swift_grpo_megatron",
    job=definitions.TrainingJob(
        image=definitions.Image(base_image=BASE_IMAGE),
        compute=definitions.Compute(
            accelerator=truss_config.AcceleratorSpec(accelerator=ACCELERATOR, count=NGPU),
            node_count=NODE_COUNT,
        ),
        runtime=definitions.Runtime(
            start_commands=["chmod +x ./run.sh && ./run.sh"],
            environment_variables={
                "HF_TOKEN": definitions.SecretReference(name="hf_access_token"),
                "WANDB_API_KEY": definitions.SecretReference(name="wandb_api_key"),
                "HF_HUB_ENABLE_HF_TRANSFER": "true",
                "CUDA_HOME": "/usr/local/cuda",
            },
            cache_config=definitions.CacheConfig(enabled=True),
            checkpointing_config=definitions.CheckpointingConfig(
                enabled=True, checkpoint_path="/tmp/checkpoints",
            ),
        ),
    ),
)
