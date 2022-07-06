# Secrets

Truss supports secrets. A model can declare names and default values of the
secrets that it needs, in config.yaml.

These secret values are provided in different ways depending on the context in
which a truss models is used.

## Local run
Truss maintains configuration information in `.truss` folder in the home
directory, in the `config.yaml` file. Secret values can be specified there for
use in local runs.

```config.yaml
secrets:
    secret_name: secret_value
```

When a truss model is run locally, the declared secrets are looked up in the
local truss config and bound from there.

## Run docker image locally
When a truss model is executed via docker image locally, secrets are again
looked up from the local truss configuration and bound from there. This binding
for docker images is done by populating secrets into a local directory and
mounting the directory onto the local container.

## Run on a model serving environment

A model serving environment can provide secrets to the docker image in following
ways:
1. Environment variables prefixed with `TRUSS_SECRET_`.

    A secret resolver in the model server resolves each declared secret name,
    say called `secret_name` by looking for `TRUSS_SECRET_secret_name`
    environment variable. For example, AWS's Secrets Manager can be used to inject
    secrets as environment variables.

2. Mounted volume on `/secrets`

    Each individual secret should be represented as a file in `/secrets` and the
    content should be the secret's value. For example, on k8s, this can be
    achieved by mounting a k8s secret.
