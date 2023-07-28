import json
import logging
import os
import sys
from functools import wraps
from pathlib import Path
from typing import Callable, Optional, Union

import rich
import rich_click as click
import truss
from truss.remote.remote_cli import inquire_model_name, inquire_remote_name
from truss.remote.remote_factory import RemoteFactory

logging.basicConfig(level=logging.INFO)


click.rich_click.COMMAND_GROUPS = {
    "truss": [
        {
            "name": "Main usage",
            "commands": ["init", "push", "watch", "predict"],
            "table_styles": {  # type: ignore
                "row_styles": ["green"],
            },
        },
        {
            "name": "Advanced Usage",
            "commands": ["image", "container", "cleanup"],
            "table_styles": {  # type: ignore
                "row_styles": ["yellow"],
            },
        },
    ]
}


def echo_output(f: Callable[..., object]):
    @wraps(f)
    def wrapper(*args, **kwargs):
        click.echo(f(*args, **kwargs))

    return wrapper


def error_handling(f: Callable[..., object]):
    @wraps(f)
    def wrapper(*args, **kwargs):
        try:
            f(*args, **kwargs)
        except Exception as e:
            click.echo(e)

    return wrapper


def print_help() -> None:
    ctx = click.get_current_context()
    click.echo(ctx.get_help())


@click.group(name="truss", invoke_without_command=True)  # type: ignore
@click.pass_context
@click.version_option(truss.version())
def truss_cli(ctx) -> None:
    """truss: The simplest way to serve models in production"""
    if not ctx.invoked_subcommand:
        click.echo(ctx.get_help())


@click.group()
def container():
    """Subcommands for truss container"""
    pass


@click.group()
def image():
    """Subcommands for truss image"""
    pass


@truss_cli.command()
@click.argument("target_directory", required=True)
@click.option(
    "-s",
    "--skip-confirm",
    is_flag=True,
    show_default=True,
    default=False,
    help="Skip confirmation prompt.",
)
@click.option(
    "-t",
    "--trainable",
    is_flag=True,
    show_default=True,
    default=False,
    help="Create a trainable truss.",
)
@error_handling
def init(target_directory, skip_confirm, trainable) -> None:
    """Create a new truss.

    TARGET_DIRECTORY: A Truss is created in this directory
    """
    tr_path = Path(target_directory)
    if skip_confirm or click.confirm(f"A Truss will be created at {tr_path}"):
        truss.init(target_directory=target_directory, trainable=trainable)
        click.echo(f"Truss was created in {tr_path}")


@image.command()  # type: ignore
@click.argument("build_dir")
@click.argument("target_directory", required=False)
@error_handling
def build_context(build_dir, target_directory: str) -> None:
    """
    Create a docker build context for a Truss.

    BUILD_DIR: Folder where image context is built for Truss

    TARGET_DIRECTORY: A Truss directory. If none, use current directory.
    """
    tr = _get_truss_from_directory(target_directory=target_directory)
    tr.docker_build_setup(build_dir=Path(build_dir))


@image.command()  # type: ignore
@click.argument("target_directory", required=False)
@click.argument("build_dir", required=False)
@error_handling
@click.option("--tag", help="Docker image tag")
def build(target_directory: str, build_dir: Path, tag) -> None:
    """
    Builds the docker image for a Truss.

    TARGET_DIRECTORY: A Truss directory. If none, use current directory.

    BUILD_DIR: Image context. If none, a temp directory is created.
    """
    tr = _get_truss_from_directory(target_directory=target_directory)
    if build_dir:
        build_dir = Path(build_dir)
    tr.build_serving_docker_image(build_dir=build_dir, tag=tag)


@image.command()  # type: ignore
@click.argument("target_directory", required=False)
@click.argument("build_dir", required=False)
@click.option("--tag", help="Docker build image tag")
@click.option("--port", type=int, default=8080, help="Local port used to run image")
@click.option(
    "--attach", is_flag=True, default=False, help="Flag for attaching the process"
)
@error_handling
def run(target_directory: str, build_dir: Path, tag, port, attach) -> None:
    """
    Runs the docker image for a Truss.

    TARGET_DIRECTORY: A Truss directory. If none, use current directory.

    BUILD_DIR: Image context. If none, a temp directory is created.
    """
    tr = _get_truss_from_directory(target_directory=target_directory)
    if build_dir:
        build_dir = Path(build_dir)
    urls = tr.get_urls_from_truss()
    if urls:
        click.confirm(
            f"Container already exists at {urls}. Are you sure you want to continue?"
        )
    tr.docker_run(build_dir=build_dir, tag=tag, local_port=port, detach=not attach)


@truss_cli.command()
@click.argument("target_directory", required=False, default=os.getcwd())
@click.option(
    "--remote",
    type=str,
    required=False,
    help="Name of the remote in .trussrc to patch changes to",
)
@error_handling
def watch(
    target_directory: str,
    remote: str,
) -> None:
    """
    Seamless remote development with truss

    TARGET_DIRECTORY: A Truss directory. If none, use current directory.
    """

    # TODO: ensure that provider support draft
    if not remote:
        remote = inquire_remote_name(RemoteFactory.get_available_config_names())

    remote_provider = RemoteFactory.create(remote=remote)

    tr = _get_truss_from_directory(target_directory=target_directory)
    model_name = tr.spec.config.model_name
    if not model_name:
        rich.print(
            "🧐 NoneType model_name provided in config.yaml. "
            "Please check that you have the correct model name in your config file."
        )
        sys.exit(1)

    rich.print(f"👀 Watching for changes to truss at '{target_directory}' ...")
    remote_provider.sync_truss_to_dev_version_by_name(model_name, target_directory)  # type: ignore


@truss_cli.command()
@click.option("--target_directory", required=False, help="Directory of truss")
@click.option(
    "--request",
    type=str,
    required=False,
    help="String formatted as json that represents request",
)
@click.option(
    "--build-dir",
    type=click.Path(exists=True),
    required=False,
    help="Directory where context is built",
)
@click.option("--tag", help="Docker build image tag")
@click.option("--port", type=int, default=8080, help="Local port used to run image")
@click.option(
    "--no-docker",
    is_flag=True,
    default=False,
    help="Flag to run prediction with a docker container",
)
@click.option(
    "--request-file",
    type=click.Path(exists=True),
    help="Path to json file containing the request",
)
@error_handling
@echo_output
def predict(
    target_directory: str,
    request: Union[bytes, str],
    build_dir,
    tag,
    port,
    no_docker,
    request_file,
):
    """
    Invokes the packaged model

    TARGET_DIRECTORY: A Truss directory. If none, use current directory.

    REQUEST: String formatted as json that represents request

    BUILD_DIR: Directory where context is built. If none, a temp directory is created.

    TAG: Docker build image tag

    PORT: Local port used to run image

    NO_DOCKER: Flag to run prediction without a docker container

    REQUEST_FILE: Path to json file containing the request
    """
    if request is not None:
        request_data = json.loads(request)
    elif request_file is not None:
        with open(request_file) as json_file:
            request_data = json.load(json_file)
    else:
        raise ValueError("At least one of request or request-file must be supplied.")

    tr = _get_truss_from_directory(target_directory=target_directory)
    if no_docker:
        return tr.server_predict(request_data)
    else:
        return tr.docker_predict(
            request_data, build_dir=build_dir, tag=tag, local_port=port, detach=True
        )


@truss_cli.command()
@click.argument("target_directory", required=False, default=os.getcwd())
@click.option(
    "--remote",
    type=str,
    required=False,
    help="Name of the remote in .trussrc to push to",
)
@click.option("--model-name", type=str, required=False, help="Name of the model")
@click.option(
    "--publish",
    type=bool,
    is_flag=True,
    required=False,
    default=False,
    help="Publish truss as production deployment.",
)
@click.option(
    "--trusted",
    type=bool,
    is_flag=True,
    required=False,
    default=False,
    help="Trust truss with hosted secrets.",
)
@error_handling
def push(
    target_directory: str,
    remote: str,
    model_name: str,
    publish: bool = False,
    trusted: bool = False,
) -> None:
    """
    Pushes a truss to a TrussRemote.

    TARGET_DIRECTORY: A Truss directory. If none, use current directory.

    """
    if not remote:
        remote = inquire_remote_name(RemoteFactory.get_available_config_names())

    remote_provider = RemoteFactory.create(remote=remote)

    tr = _get_truss_from_directory(target_directory=target_directory)

    # Push
    model_name = model_name or tr.spec.config.model_name
    if not model_name:
        model_name = inquire_model_name()

    # Write model name to config if it's not already there
    if model_name != tr.spec.config.model_name:
        tr.spec.config.model_name = model_name
        tr.spec.config.write_to_yaml_file(tr.spec.config_path, verbose=False)

    # TODO(Abu): This needs to be refactored to be more generic
    _ = remote_provider.push(tr, model_name, publish=publish, trusted=trusted)  # type: ignore

    click.echo(f"Model {model_name} was successfully pushed.")


@container.command()  # type: ignore
@click.argument("target_directory", required=False)
@error_handling
def logs(target_directory) -> None:
    """
    Get logs in a container is running for a truss

    TARGET_DIRECTORY: A Truss directory. If none, use current directory.
    """
    for log in _get_truss_from_directory(
        target_directory=target_directory
    ).serving_container_logs():
        click.echo(log)


@container.command()  # type: ignore
@click.argument("target_directory", required=False)
def kill(target_directory: str) -> None:
    """
    Kills containers related to truss.

    TARGET_DIRECTORY: A Truss directory. If none, use current directory.
    """
    tr = _get_truss_from_directory(target_directory=target_directory)
    tr.kill_container()


@container.command()  # type: ignore
def kill_all() -> None:
    "Kills all truss containers that are not manually persisted"
    truss.kill_all()


@truss_cli.command()
@error_handling
def cleanup() -> None:
    """
    Clean up truss data.

    Truss creates temporary directories for various operations
    such as for building docker images. This command clears
    that data to free up disk space.
    """
    truss.build.cleanup()


def _get_truss_from_directory(target_directory: Optional[str] = None):
    """Gets Truss from directory. If none, use the current directory"""
    if target_directory is None:
        target_directory = os.getcwd()
    return truss.load(target_directory)


truss_cli.add_command(container)
truss_cli.add_command(image)

if __name__ == "__main__":
    truss_cli()
