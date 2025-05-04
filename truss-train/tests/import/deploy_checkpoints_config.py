from truss_train import definitions
from truss.base import truss_config

deploy_checkpoint = definitions.DeployCheckpointsConfig(
    compute=definitions.Compute(
        accelerator=truss_config.AcceleratorSpec(
            accelerator=truss_config.Accelerator.A10G,
            count=4,
        )
    ),
    runtime=definitions.CheckpointDeployRuntime(
        environment_variables={
            "HF_TOKEN": definitions.SecretReference(name="hf_access_token"),
        },
    ),
    checkpoint_details=definitions.CheckpointDetails(
        base_model_id="unsloth/gemma-3-1b-it",
        checkpoints=[
            definitions.Checkpoint(id="checkpoint-24", name="checkpoint-24", training_job_id="lqz4pw4"),
            definitions.Checkpoint(id="checkpoint-42", name="checkpoint-42", training_job_id="lqz4pw4"),
        ],
    ),
)