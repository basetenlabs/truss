from truss.base import truss_config
from truss_train import (
    CheckpointList,
    Compute,
    DeployCheckpointsConfig,
    DeployCheckpointsRuntime,
    FullCheckpoint,
    LoRACheckpoint,
    LoRADetails,
    ModelWeightsFormat,
    SecretReference,
)

deploy_checkpoint = DeployCheckpointsConfig(
    compute=Compute(
        accelerator=truss_config.AcceleratorSpec(
            accelerator=truss_config.Accelerator.A10G, count=4
        )
    ),
    runtime=DeployCheckpointsRuntime(
        environment_variables={"HF_TOKEN": SecretReference(name="hf_access_token")}
    ),
    checkpoint_details=CheckpointList(
        base_model_id="unsloth/gemma-3-1b-it",
        checkpoints=[
            LoRACheckpoint(
                training_job_id="lqz4pw4",
                paths=["lqz4pw4/rank-0/checkpoint-24/"],
                lora_details=LoRADetails(rank=16),
                model_weight_format=ModelWeightsFormat.LORA,  # Not required, have it purely to test the import
            ),
            LoRACheckpoint(
                training_job_id="lqz4pw5",
                paths=["lqz4pw5/rank-0/checkpoint-42/"],
                lora_details=LoRADetails(rank=8),
            ),
            FullCheckpoint(
                training_job_id="lqz4pw6",
                paths=["lqz4pw6/checkpoint-123/"],
                model_weight_format=ModelWeightsFormat.FULL,  # Not required, have it purely to test the import
            ),
        ],
    ),
)
