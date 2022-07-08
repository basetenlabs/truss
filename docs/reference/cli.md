# Truss command line reference

You can always review these commands with `python -m truss --help`, which will print a complete list of commands.

### build-context

Create a docker build context for a scaffold.

### build-image

Builds the docker image for a scaffold.

### init

Initializes a empty scaffold directory

#### Args:

**dir** (required): A new directory to create the Truss in


#### Examples:

```bash
python -m truss init my_model
```

```bash
python -m truss init iris_rfc
```

### predict

Runs prediction for a scaffold in a docker image or locally


### run-example

Runs examples specified in the scaffold over docker.


### run-image

Runs the docker image for a scaffold.
