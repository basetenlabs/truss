from truss.base import truss_config
from truss_train import (
    CheckpointList,
    Compute,
    DeployCheckpointsConfig,
    DeployCheckpointsRuntime,
    FullCheckpoint,
    Image,
    LoRACheckpoint,
    LoRADetails,
    Runtime,
    SecretReference,
    TrainingJob,
    TrainingProject,
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
                checkpoint_name="checkpoint-24",
                lora_details=LoRADetails(rank=16),
            ),
            LoRACheckpoint(
                training_job_id="lqz4pw5",
                checkpoint_name="checkpoint-42",
                lora_details=LoRADetails(rank=8),
            ),
            FullCheckpoint(training_job_id="lqz4pw6", checkpoint_name="checkpoint-123"),
        ],
    ),
)

runtime_config = Runtime(
    start_commands=["/bin/bash ./my-entrypoint.sh"],
    environment_variables={
        "FOO_VAR": "FOO_VAL",
        "BAR_VAR": SecretReference(name="BAR_SECRET"),
    },
)

training_job = TrainingJob(
    image=Image(base_image="base-image"),
    compute=Compute(node_count=1, cpu_count=4),
    runtime=runtime_config,
)

first_project = TrainingProject(name="first-project", job=training_job)
