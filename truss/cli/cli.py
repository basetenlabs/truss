import inspect
import json
import logging
import os
import sys
from functools import wraps
from pathlib import Path
from typing import Callable, Optional

import rich
import rich_click as click
import truss
from truss.cli.create import ask_name, select_server_backend
from truss.remote.baseten.core import (
    ModelId,
    ModelIdentifier,
    ModelName,
    ModelVersionId,
)
from truss.remote.baseten.service import BasetenService
from truss.remote.remote_cli import inquire_model_name, inquire_remote_name
from truss.remote.remote_factory import USER_TRUSSRC_PATH, RemoteFactory
from truss.truss_config import ModelServer

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

console = rich.console.Console()


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
        except click.UsageError as e:
            raise e  # You can re-raise the exception or handle it different
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
    "-b",
    "--backend",
    show_default=True,
    default=ModelServer.TrussServer.value,
    type=click.Choice([server.value for server in ModelServer]),
)
@error_handling
def init(target_directory, backend) -> None:
    """Create a new truss.

    TARGET_DIRECTORY: A Truss is created in this directory
    """
    if os.path.isdir(target_directory):
        raise click.ClickException(
            f'Error: Directory "{target_directory}" already exists and cannot be overwritten.'
        )
    tr_path = Path(target_directory)
    build_config = select_server_backend(ModelServer[backend])
    model_name = ask_name()
    truss.init(
        target_directory=target_directory,
        build_config=build_config,
        model_name=model_name,
    )
    click.echo(f"Truss {model_name} was created in {tr_path.absolute()}")


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
@click.option(
    "--logs",
    is_flag=True,
    show_default=True,
    default=False,
    help="Automatically open remote logs tab",
)
@error_handling
def watch(
    target_directory: str,
    remote: str,
    logs: bool,
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

    service = remote_provider.get_service(model_identifier=ModelName(model_name))
    logs_url = remote_provider.get_remote_logs_url(service)
    rich.print(f"🪵  View logs for your deployment at {logs_url}")
    remote_provider.sync_truss_to_dev_version_by_name(model_name, target_directory)


def _extract_and_validate_model_identifier(
    target_directory: str,
    model_id: Optional[str],
    model_version_id: Optional[str],
    published: Optional[bool],
) -> ModelIdentifier:
    if published and (model_id or model_version_id):
        raise click.UsageError(
            "Cannot use --published with --model or --model-version."
        )

    model_identifier: ModelIdentifier
    if model_version_id:
        model_identifier = ModelVersionId(model_version_id)
    elif model_id:
        model_identifier = ModelId(model_id)
    else:
        tr = _get_truss_from_directory(target_directory=target_directory)
        model_name = tr.spec.config.model_name
        if not model_name:
            raise click.UsageError("Truss config is missing a model name.")
        model_identifier = ModelName(model_name)
    return model_identifier


def _extract_request_data(data: Optional[str], file: Optional[Path]):
    try:
        if data is not None:
            return json.loads(data)
        if file is not None:
            return json.loads(Path(file).read_text())
    except json.JSONDecodeError:
        raise click.UsageError("Request data must be valid json.")

    raise click.UsageError(
        "You must provide exactly one of '--data (-d)' or '--file (-f)' options."
    )


@truss_cli.command()
@click.option("--target-directory", required=False, help="Directory of truss")
@click.option(
    "--remote",
    type=str,
    required=False,
    help="Name of the remote in .trussrc to push to",
)
@click.option(
    "-d",
    "--data",
    type=str,
    required=False,
    help="String formatted as json that represents request",
)
@click.option(
    "-f",
    "--file",
    type=click.Path(exists=True),
    help="Path to json file containing the request",
)
@click.option(
    "--published",
    type=bool,
    is_flag=True,
    required=False,
    default=False,
    help="Call the published model deployment.",
)
@click.option(
    "--model-version",
    type=str,
    required=False,
    help="[DEPRECATED] Use --model-deployment instead, this will be removed in future release. ID of model deployment",
)
@click.option(
    "--model-deployment",
    type=str,
    required=False,
    help="ID of model deployment to call",
)
@click.option(
    "--model",
    type=str,
    required=False,
    help="ID of model to call",
)
@echo_output
def predict(
    target_directory: str,
    remote: str,
    data: Optional[str],
    file: Optional[Path],
    published: Optional[bool],
    model_version: Optional[str],
    model_deployment: Optional[str],
    model: Optional[str],
):
    """
    Calls the packaged model

    TARGET_DIRECTORY: A Truss directory. If none, use current directory.

    REQUEST: String formatted as json that represents request

    REQUEST_FILE: Path to json file containing the request
    """
    if not remote:
        remote = inquire_remote_name(RemoteFactory.get_available_config_names())

    remote_provider = RemoteFactory.create(remote=remote)

    if model_version:
        console.print(
            "[DEPRECATED] --model-version is deprecated, use --model-deployment instead.",
            style="yellow",
        )
        model_deployment = model_version

    model_identifier = _extract_and_validate_model_identifier(
        target_directory,
        model_id=model,
        model_version_id=model_deployment,
        published=published,
    )

    request_data = _extract_request_data(data=data, file=file)

    service = remote_provider.get_service(
        model_identifier=model_identifier, published=published
    )

    # Log deployment ID for Baseten models.
    if isinstance(service, BasetenService):
        rich.print(
            f"Calling predict on {'[cyan]development[/cyan] ' if service.is_draft else ''}"
            f"deployment ID {service.model_version_id}..."
        )

    result = service.predict(request_data)
    if inspect.isgenerator(result):
        for chunk in result:
            click.echo(chunk, nl=False)
        return
    rich.print_json(data=result)


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
    help=(
        "Push the truss as a published deployment. If no production "
        "deployment exists, promote the truss to production."
    ),
)
@click.option(
    "--trusted",
    type=bool,
    is_flag=True,
    required=False,
    default=False,
    help="Trust truss with hosted secrets.",
)
@click.option(
    "--promote",
    type=bool,
    is_flag=True,
    required=False,
    default=False,
    help="After deploy completes, promotes the truss to production.",
)
@error_handling
def push(
    target_directory: str,
    remote: str,
    model_name: str,
    publish: bool = False,
    trusted: bool = False,
    promote: bool = False,
) -> None:
    """
    Pushes a truss to a TrussRemote.

    TARGET_DIRECTORY: A Truss directory. If none, use current directory.

    """
    if not remote:
        remote = inquire_remote_name(RemoteFactory.get_available_config_names())

    remote_provider = RemoteFactory.create(remote=remote)

    tr = _get_truss_from_directory(target_directory=target_directory)

    model_name = model_name or tr.spec.config.model_name
    if not model_name:
        model_name = inquire_model_name()

    # Write model name to config if it's not already there
    if model_name != tr.spec.config.model_name:
        tr.spec.config.model_name = model_name
        tr.spec.config.write_to_yaml_file(tr.spec.config_path, verbose=False)

    # TODO(Abu): This needs to be refactored to be more generic
    service = remote_provider.push(tr, model_name, publish=publish, trusted=trusted, promote=promote)  # type: ignore

    click.echo(f"✨ Model {model_name} was successfully pushed ✨")

    if service.is_draft:
        draft_model_text = """
|---------------------------------------------------------------------------------------|
| Your model has been deployed as a development model. Development models allow you to  |
| iterate quickly during the deployment process.                                        |
|                                                                                       |
| When you are ready to publish your deployed model as a new deployment,                |
| pass `--publish` to the `truss push` command. To monitor changes to your model and    |
| rapidly iterate, run the `truss watch` command.                                       |
|                                                                                       |
|---------------------------------------------------------------------------------------|
"""

        click.echo(draft_model_text)

    # Log a warning if using secrets without --trusted.
    # TODO(helen): this could be moved to a separate function that includes more config checks.
    if tr.spec.config.secrets and not trusted:
        not_trusted_text = """Warning: your Truss has secrets but was not pushed with --trusted.
Please push with --trusted to grant access to secrets.
"""
        console.print(not_trusted_text, style="red")

    logs_url = remote_provider.get_remote_logs_url(service)  # type: ignore[attr-defined]
    rich.print(f"🪵  View logs for your deployment at {logs_url}")


@truss_cli.command()
def configure():
    # Read the original file content
    with open(USER_TRUSSRC_PATH, "r") as f:
        original_content = f.read()

    # Open the editor and get the modified content
    edited_content = click.edit(original_content)

    # If the content was modified, save it
    if edited_content is not None and edited_content != original_content:
        with open(USER_TRUSSRC_PATH, "w") as f:
            f.write(edited_content)
            click.echo(f"Changes saved to {USER_TRUSSRC_PATH}")
    else:
        click.echo("No changes made.")


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
