from truss_train import definitions

runtime_config = definitions.Runtime(
    entrypoint=["/bin/bash ./my-entrypoint.sh"],
    environment_variables={
        "FOO_VAR": "FOO_VAL",
        "BAR_VAR": definitions.SecretReference(name="BAR_SECRET"),
    },
)

_ = definitions.TrainingJob(
    image=definitions.Image(base_image="base-image"),
    compute=definitions.Compute(node_count=1, cpu_count=1),
    runtime=runtime_config,
)
