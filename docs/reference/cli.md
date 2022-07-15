# Truss command line reference

Use the Truss package via the command line to package and serve models.

Usage: `truss [OPTIONS] COMMAND [ARGS]`

## Options

#### --help

You can always review these commands with `python -m truss --help`, which will print a complete list of commands.

## Commands

#### build_context

Create a docker build context for a Truss.

Args:
* BUILD_DIR: Folder where image context is built for Truss
* TARGET DIRECTORY: A Truss directory. If none, use current directory.

#### build_image

Builds the docker image for a Truss.

Args: 
* TARGET DIRECTORY: A Truss directory. If none, use current directory.
* BUILD_DIR: Image context. If none, a temp directory is created.

#### cleanup

Clean up truss data.
    
Truss creates temporary directories for various operations
such as for building docker images. This command clears
that data to free up disk space.

#### cli_group

#### get_container_logs

Get logs in a container is running for a truss

Args:
* TARGET DIRECTORY: A Truss directory. If none, use current directory.

#### init

Initializes an empty Truss directory.

Args:
* TARGET_DIRECTORY: A Truss is created in this directory

#### kill

Kills containers related to truss.

Args:
* TARGET DIRECTORY: A Truss directory. If none, use current directory.

#### kill_all

Kills all truss containers that are not manually persisted

#### predict

Runs prediction for a Truss in a docker image or locally

#### run_example

Runs examples specified in the Truss, over docker.

Args:
* TARGET DIRECTORY: A Truss directory. If none, use current directory.

#### run_image

Runs the docker image for a Truss.

Args:
* TARGET DIRECTORY: A Truss directory. If none, use current directory.
* BUILD_DIR: Image context. If none, a temp directory is created.