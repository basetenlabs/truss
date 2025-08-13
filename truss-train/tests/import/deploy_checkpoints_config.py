from truss.base import truss_config
from truss_train import definitions

deploy_checkpoint = definitions.DeployCheckpointsConfig(
    compute=definitions.Compute(
        accelerator=truss_config.AcceleratorSpec(
            accelerator=truss_config.Accelerator.A10G, count=4
        )
    ),
    runtime=definitions.DeployCheckpointsRuntime(
        environment_variables={
            "HF_TOKEN": definitions.SecretReference(name="hf_access_token")
        }
    ),
    checkpoint_details=definitions.CheckpointList(
        base_model_id="unsloth/gemma-3-1b-it",
        checkpoints=[
            definitions.LoRACheckpoint(
                training_job_id="lqz4pw4",
                paths=["lqz4pw4/rank-0/checkpoint-24/"],
                lora_details=definitions.LoRADetails(rank=16),
            ),
            definitions.LoRACheckpoint(
                training_job_id="lqz4pw5",
                paths=["lqz4pw5/rank-0/checkpoint-42/"],
                lora_details=definitions.LoRADetails(rank=8),
            ),
        ],
    ),
)
