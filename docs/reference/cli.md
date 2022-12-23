# Truss command line reference

Use the Truss package via the command line to package and serve models.

Usage: `truss [OPTIONS] COMMAND [ARGS]`

## Options

#### --help

You can always review these commands with `truss --help`, which will print a complete list of commands.

#### -v, --version

Show Truss package version.

## Commands

#### build-context

Create a docker build context for a Truss.

Args:
* BUILD_DIR: Folder where image context is built for Truss
* TARGET_DIRECTORY: A Truss directory. If none, use current directory.

#### build-image

Builds the docker image for a Truss.

Args:
* TARGET_DIRECTORY: A Truss directory. If none, use current directory.
* BUILD_DIR: Image context. If none, a temp directory is created.

#### cleanup

Clean up truss data.

Truss creates temporary directories for various operations
such as for building docker images. This command clears
that data to free up disk space.

#### get-container-logs

Get logs in a container is running for a truss

Args:
* TARGET_DIRECTORY: A Truss directory. If none, use current directory.

#### init

Initializes an empty Truss directory.

Args:
* TARGET_DIRECTORY: A Truss is created in this directory

#### kill

Kills containers related to truss.

Args:
* TARGET_DIRECTORY: A Truss directory. If none, use current directory.

#### kill-all

Kills all truss containers that are not manually persisted

#### predict

Invokes the packaged model, either locally or in a Docker container.

Args:

* TARGET_DIRECTORY: A Truss directory. If none, use current directory.
* REQUEST: String formatted as json that represents request
* BUILD_DIR: Directory where context is built. If none, a temp directory is created.
* TAG: Docker build image tag
* PORT: Local port used to run image
* RUN_LOCAL: Flag to run prediction locally (instead of on Docker)
* REQUEST_FILE: Path to json file containing the request

#### run-example

Runs examples specified in the Truss, over docker.

Args:
* TARGET_DIRECTORY: A Truss directory. If none, use current directory.

#### run-image

Runs the docker image for a Truss.

Args:
* TARGET_DIRECTORY: A Truss directory. If none, use current directory.
* BUILD_DIR: Image context. If none, a temp directory is created.
