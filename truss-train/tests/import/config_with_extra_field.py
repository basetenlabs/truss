from truss_train import definitions

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
    extra_field="this_should_fail",  # type: ignore
)

first_project = definitions.TrainingProject(name="first-project", job=training_job)
