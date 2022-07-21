import json
import logging
import os
from functools import wraps
from pathlib import Path

import click

import truss

logging.basicConfig(level=logging.INFO)


def echo_output(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        click.echo(f(*args, **kwargs))
    return wrapper


def error_handling(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        try:
            f(*args, **kwargs)
        except ValueError as e:
            click.echo(e)
            print_help()
        except Exception as e:
            click.echo(e)
    return wrapper


def print_help():
    ctx = click.get_current_context()
    click.echo(ctx.get_help())


@click.group(name="truss")
def cli_group():
    pass


@cli_group.command()
@click.argument("target_directory", required=True)
@click.option("-s", "--skip-confirm", is_flag=True, show_default=True, default=False, help="Skip confirmation prompt.")
@error_handling
def init(target_directory, skip_confirm):
    """ Initializes an empty Truss directory.

    TARGET_DIRECTORY: A Truss is created in this directory
    """
    tr_path = Path(target_directory)
    if skip_confirm or click.confirm(f'A Truss will be created at {tr_path}'):
        truss.init(target_directory=target_directory)
        click.echo(f"Truss was created in {tr_path}")


@cli_group.command()
@click.argument("build_dir")
@click.argument("target_directory", required=False)
@error_handling
def build_context(build_dir, target_directory) -> None:
    """
    Create a docker build context for a Truss.

    BUILD_DIR: Folder where image context is built for Truss

    TARGET DIRECTORY: A Truss directory. If none, use current directory.
    """
    tr = _get_truss_from_directory(target_directory=target_directory)
    tr.docker_build_setup(build_dir=Path(build_dir))


@cli_group.command()
@click.argument("target_directory", required=False)
@click.argument("build_dir", required=False)
@error_handling
@click.option("--tag", help="Docker image tag")
def build_image(target_directory, build_dir, tag):
    """
    Builds the docker image for a Truss.

    TARGET DIRECTORY: A Truss directory. If none, use current directory.

    BUILD_DIR: Image context. If none, a temp directory is created.
    """
    tr = _get_truss_from_directory(target_directory=target_directory)
    if build_dir:
        build_dir = Path(build_dir)
    tr.build_docker_image(build_dir=build_dir, tag=tag)


@cli_group.command()
@click.argument("target_directory", required=False)
@click.argument("build_dir", required=False)
@click.option("--tag", help="Docker build image tag")
@click.option("--port", type=int, default=8080, help="Local port used to run image")
@click.option(
    "--detach", is_flag=True, default=False, help="Flag for detaching process"
)
@error_handling
def run_image(target_directory, build_dir, tag, port, detach):
    """
    Runs the docker image for a Truss.

    TARGET DIRECTORY: A Truss directory. If none, use current directory.

    BUILD_DIR: Image context. If none, a temp directory is created.
    """
    tr = _get_truss_from_directory(target_directory=target_directory)
    if build_dir:
        build_dir = Path(build_dir)
    urls = tr.get_urls_from_truss()
    if urls:
        click.confirm(
            f"Container already exists at {urls}. Are you sure you want to continue?")
    tr.docker_run(build_dir=build_dir, tag=tag, local_port=port, detach=detach)


@cli_group.command()
@click.option("--target_directory", required=False, help="Directory of truss")
@click.option("--request", type=str, required=False, help="String formatted as json that represents request")
@click.option("--build-dir", type=click.Path(exists=True), required=False, help="Directory where context is built")
@click.option("--tag", help="Docker build image tag")
@click.option("--port", type=int, default=8080, help="Local port used to run image")
@click.option("--run-local", is_flag=True, default=False, help="Flag to run prediction locally")
@click.option("--request-file", type=click.Path(exists=True), help="Path to json file containing the request")
@error_handling
@echo_output
def predict(target_directory, request, build_dir, tag, port, run_local, request_file):
    """Runs prediction for a Truss in a docker image or locally """
    if request is not None:
        request_data = json.loads(request)
    elif request_file is not None:
        with open(request_file) as json_file:
            request_data = json.load(json_file)
    else:
        raise ValueError(
            "At least one of request or request-file must be supplied.")

    tr = _get_truss_from_directory(target_directory=target_directory)
    if run_local:
        return tr.server_predict(request_data)
    else:
        return tr.docker_predict(
            request_data, build_dir=build_dir,
            tag=tag, local_port=port, detach=True
        )


@cli_group.command()
@click.argument("target_directory", required=False)
@click.option("--name", type=str, required=False, help="Name of example to run")
@click.option(
    "--local", is_flag=True, default=False, help="Flag to run prediction locally"
)
@error_handling
@echo_output
def run_example(target_directory, name, local):
    """
    Runs examples specified in the Truss, over docker.

    TARGET DIRECTORY: A Truss directory. If none, use current directory.
    """
    tr = _get_truss_from_directory(target_directory=target_directory)
    predict_fn = tr.docker_predict
    if local:
        predict_fn = tr.server_predict

    if name is not None:
        example = tr.example(name)
        click.echo(f'Running example: {name}')
        return predict_fn(example)
    else:
        example_outputs = []
        for name, example in tr.examples().items():
            click.echo(f'Running example: {name}')
            example_outputs.append(predict_fn(example))
        return example_outputs


@cli_group.command()
@click.argument("target_directory", required=False)
@error_handling
def get_container_logs(target_directory):
    """
    Get logs in a container is running for a truss

    TARGET DIRECTORY: A Truss directory. If none, use current directory.
    """
    for log in _get_truss_from_directory(
        target_directory=target_directory
    ).container_logs():
        click.echo(log)


@cli_group.command()
@click.argument("target_directory", required=False)
def kill(target_directory):
    """
    Kills containers related to truss.

    TARGET DIRECTORY: A Truss directory. If none, use current directory.
    """
    tr = _get_truss_from_directory(target_directory=target_directory)
    tr.kill_container()


@cli_group.command()
def kill_all():
    "Kills all truss containers that are not manually persisted"
    truss.kill_all()


@cli_group.command()
@error_handling
def cleanup() -> None:
    """
    Clean up truss data.

    Truss creates temporary directories for various operations
    such as for building docker images. This command clears
    that data to free up disk space.
    """
    truss.build.cleanup()


def _get_truss_from_directory(target_directory: str = None):
    """Gets Truss from directory. If none, use the current directory"""
    if target_directory is None:
        target_directory = os.getcwd()
    return truss.from_directory(target_directory)


if __name__ == "__main__":
    cli_group()
