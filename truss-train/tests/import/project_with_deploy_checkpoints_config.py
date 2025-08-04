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
                training_job_id="waqeqweq", path_details=[], lora_rank=16
            ),
            definitions.LoRACheckpoint(
                training_job_id="waqeqweqs", path_details=[], lora_rank=8
            ),
        ],
    ),
)

runtime_config = definitions.Runtime(
    start_commands=["/bin/bash ./my-entrypoint.sh"],
    environment_variables={
        "FOO_VAR": "FOO_VAL",
        "BAR_VAR": definitions.SecretReference(name="BAR_SECRET"),
    },
)

training_job = definitions.TrainingJob(
    image=definitions.Image(base_image="base-image"),
    compute=definitions.Compute(node_count=1, cpu_count=4),
    runtime=runtime_config,
)

first_project = definitions.TrainingProject(name="first-project", job=training_job)
