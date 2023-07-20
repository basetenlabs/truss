import json
import logging
import os
from functools import wraps
from pathlib import Path
from typing import Callable, List, Optional, Union

import click
import truss
import yaml
from truss.remote.remote_factory import RemoteFactory

logging.basicConfig(level=logging.INFO)


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
@click.option(
    "-v",
    "--version",
    is_flag=True,
    show_default=False,
    default=False,
    help="Show Truss package version.",
)
def cli_group(ctx, version) -> None:
    if not ctx.invoked_subcommand:
        if version:
            click.echo(truss.version())
        else:
            click.echo(ctx.get_help())


@cli_group.command()
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
    """Initializes an empty Truss directory.

    TARGET_DIRECTORY: A Truss is created in this directory
    """
    tr_path = Path(target_directory)
    if skip_confirm or click.confirm(f"A Truss will be created at {tr_path}"):
        truss.init(target_directory=target_directory, trainable=trainable)
        click.echo(f"Truss was created in {tr_path}")


@cli_group.command()
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


@cli_group.command()  # type: ignore
@click.argument("target_directory", required=False)
@click.argument("build_dir", required=False)
@error_handling
@click.option("--tag", help="Docker image tag")
def build_image(target_directory: str, build_dir: Path, tag) -> None:
    """
    Builds the docker image for a Truss.

    TARGET_DIRECTORY: A Truss directory. If none, use current directory.

    BUILD_DIR: Image context. If none, a temp directory is created.
    """
    tr = _get_truss_from_directory(target_directory=target_directory)
    if build_dir:
        build_dir = Path(build_dir)
    tr.build_serving_docker_image(build_dir=build_dir, tag=tag)


@cli_group.command()
@click.argument("target_directory", required=False)
@click.argument("build_dir", required=False)
@click.option("--tag", help="Docker build image tag")
@click.option("--port", type=int, default=8080, help="Local port used to run image")
@click.option(
    "--attach", is_flag=True, default=False, help="Flag for attaching the process"
)
@error_handling
def run_image(target_directory: str, build_dir: Path, tag, port, attach) -> None:
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


@cli_group.command()
@click.argument("target_directory", required=False, default=os.getcwd())
@click.option(
    "--remote",
    type=str,
    required=True,
    help="Name of the remote in .trussrc to patch changes to",
)
@error_handling
def watch(
    target_directory: str,
    remote: str,
) -> None:
    """
    Watches local truss directory for changes and sends patch requests to remote development truss

    TARGET_DIRECTORY: A Truss directory. If none, use current directory.
    """

    # TODO: ensure that provider support draft
    remote_provider = RemoteFactory.create(remote=remote)

    tr = _get_truss_from_directory(target_directory=target_directory)
    model_name = tr.spec.config.model_name
    if not model_name:
        raise ValueError("'NoneType' model_name value provided in config.yaml")

    click.echo(f"Watching for changes to truss at: {target_directory} ...")
    remote_provider.sync_truss_to_dev_version_by_name(model_name, target_directory)  # type: ignore


@cli_group.command()
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
    Invokes the packaged model, either locally or in a Docker container.

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


@cli_group.command()
@click.option("--target_directory", required=False, help="Directory of truss")
@click.option(
    "--build-dir",
    type=click.Path(exists=True),
    required=False,
    help="Directory where context is built",
)
@click.option("--tag", help="Docker build image tag")
@click.option(
    "--var",
    multiple=True,
    help="""Training variables in key=value form where value is string.
    For more complex values use vars_yaml_file""",
)
@click.option(
    "--vars_yaml_file",
    required=False,
    help="Training variables from a yaml file",
)
@click.option(
    "--local",
    is_flag=True,
    default=False,
    help="Flag to run training locally (not on docker)",
)
@error_handling
@echo_output
def train(target_directory: str, build_dir, tag, var: List[str], vars_yaml_file, local):
    """Runs prediction for a Truss in a docker image or locally"""
    tr = _get_truss_from_directory(target_directory=target_directory)
    if vars_yaml_file is not None:
        with Path(vars_yaml_file).open() as vars_file:
            variables = yaml.safe_load(vars_file)
    else:
        variables = _variables_dict_from_option(var)
    if local:
        return tr.local_train(variables=variables)

    return tr.docker_train(build_dir=build_dir, tag=tag, variables=variables)


@cli_group.command()
@click.argument("target_directory", required=False, default=os.getcwd())
@click.option(
    "--remote",
    type=str,
    required=True,
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
@error_handling
def push(
    target_directory: str,
    remote: str,
    model_name: str,
    publish: bool = False,
) -> None:
    """
    Pushes a truss to a TrussRemote.

    TARGET_DIRECTORY: A Truss directory. If none, use current directory.

    """
    remote_provider = RemoteFactory.create(remote=remote)

    tr = _get_truss_from_directory(target_directory=target_directory)

    # Push
    model_name = model_name or tr.spec.config.model_name
    if model_name is None:
        raise ValueError(
            "Model name must be provided either as a flag or in the Truss config"
        )

    # Write model name to config if it's not already there
    if model_name != tr.spec.config.model_name:
        tr.spec.config.model_name = model_name
        tr.spec.config.write_to_yaml_file(tr.spec.config_path, verbose=False)

    # TODO(Abu): This needs to be refactored to be more generic
    _ = remote_provider.push(tr, model_name, publish=publish)  # type: ignore

    click.echo(f"Model {model_name} was successfully pushed.")


@cli_group.command()
@click.argument("target_directory", required=False)
@click.option("--name", type=str, required=False, help="Name of example to run")
@click.option(
    "--local", is_flag=True, default=False, help="Flag to run prediction locally"
)
@error_handling
@echo_output
def run_example(target_directory: str, name, local):
    """
    Runs examples specified in the Truss, over docker.

    TARGET_DIRECTORY: A Truss directory. If none, use current directory.
    """
    tr = _get_truss_from_directory(target_directory=target_directory)
    predict_fn = tr.docker_predict
    if local:
        predict_fn = tr.server_predict

    if name is not None:
        example = tr.example(name)
        click.echo(f"Running example: {name}")
        return predict_fn(example.input)
    else:
        example_outputs = []
        for example in tr.examples():
            click.echo(f"Running example: {example.name}")
            example_outputs.append(predict_fn(example.input))
        return example_outputs


@cli_group.command()
@click.argument("target_directory", required=False)
@error_handling
def get_container_logs(target_directory) -> None:
    """
    Get logs in a container is running for a truss

    TARGET_DIRECTORY: A Truss directory. If none, use current directory.
    """
    for log in _get_truss_from_directory(
        target_directory=target_directory
    ).serving_container_logs():
        click.echo(log)


@cli_group.command()  # type: ignore
@click.argument("target_directory", required=False)
def kill(target_directory: str) -> None:
    """
    Kills containers related to truss.

    TARGET_DIRECTORY: A Truss directory. If none, use current directory.
    """
    tr = _get_truss_from_directory(target_directory=target_directory)
    tr.kill_container()


@cli_group.command()
def kill_all() -> None:
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


def _get_truss_from_directory(target_directory: Optional[str] = None):
    """Gets Truss from directory. If none, use the current directory"""
    if target_directory is None:
        target_directory = os.getcwd()
    return truss.load(target_directory)


def _variables_dict_from_option(variables_list: List[str]) -> dict:
    vars_dict = {}
    for var in variables_list:
        first_equals_pos = var.find("=")
        if first_equals_pos == -1:
            raise ValueError(
                f"Training variable expected in `key=value` from but found `{var}`",
            )
        var_name = var[:first_equals_pos]
        var_value = var[first_equals_pos + 1 :]
        vars_dict[var_name] = var_value
    return vars_dict


if __name__ == "__main__":
    cli_group()
