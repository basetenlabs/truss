import inspect
import json
import logging
import os
import sys
import time
from functools import wraps
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import rich
import rich.live
import rich.spinner
import rich.table
import rich_click as click
from InquirerPy import inquirer

import truss
from truss.cli.console import console
from truss.cli.create import ask_name
from truss.constants import TRTLLM_MIN_MEMORY_REQUEST_GI
from truss.remote.baseten.core import (
    ACTIVE_STATUS,
    DEPLOYING_STATUSES,
    ModelId,
    ModelIdentifier,
    ModelName,
    ModelVersionId,
)
from truss.remote.baseten.service import BasetenService
from truss.remote.baseten.utils.status import get_displayable_status
from truss.remote.remote_cli import inquire_model_name, inquire_remote_name
from truss.remote.remote_factory import USER_TRUSSRC_PATH, RemoteFactory
from truss.truss_config import Build, ModelServer
from truss.util.config_checks import (
    check_and_update_memory_for_trt_llm_builder,
    check_secrets_for_trt_llm_builder,
)
from truss.util.errors import RemoteNetworkError

rich.spinner.SPINNERS["deploying"] = {"interval": 500, "frames": ["👾 ", " 👾"]}
rich.spinner.SPINNERS["building"] = {"interval": 500, "frames": ["🛠️ ", " 🛠️"]}
rich.spinner.SPINNERS["loading"] = {"interval": 500, "frames": ["⏱️ ", " ⏱️"]}
rich.spinner.SPINNERS["active"] = {"interval": 500, "frames": ["💚 ", " 💚"]}
rich.spinner.SPINNERS["failed"] = {"interval": 500, "frames": ["😤 ", " 😤"]}

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
        {
            "name": "Chains",
            "commands": ["chains"],
            "table_styles": {  # type: ignore
                "row_styles": ["red"],
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
        except click.UsageError as e:
            raise e  # You can re-raise the exception or handle it different
        except Exception as e:
            click.secho(f"ERROR ({type(e)}: {e}", fg="red")

    return wrapper


_HUMANFRIENDLY_LOG_LEVEL = "humanfriendly"
_log_level_str_to_level = {
    _HUMANFRIENDLY_LOG_LEVEL: logging.INFO,
    "I": logging.INFO,
    "INFO": logging.INFO,
    "D": logging.DEBUG,
    "DEBUG": logging.DEBUG,
}


def _set_logging_level(log_level: str) -> None:
    level = _log_level_str_to_level[log_level]
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    if log_level == _HUMANFRIENDLY_LOG_LEVEL:
        formatter = logging.Formatter(fmt="%(message)s")
    else:
        # Absl-inspired logging for technical output.
        log_format = "%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s"
        date_format = "%m%d %H:%M:%S"
        formatter = logging.Formatter(fmt=log_format, datefmt=date_format)

    if root_logger.handlers:
        for handler in root_logger.handlers:
            handler.setFormatter(formatter)
    else:
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)


def log_level_option(f):
    def callback(ctx, param, value):
        _set_logging_level(value)
        return value

    return click.option(
        "--log",
        default=_HUMANFRIENDLY_LOG_LEVEL,
        expose_value=False,
        help="Customizes logging.",
        type=click.Choice(list(_log_level_str_to_level.keys()), case_sensitive=False),
        callback=callback,
    )(f)


def _format_link(text: str) -> str:
    return f"[link={text}]{text}[/link]"


def print_help() -> None:
    ctx = click.get_current_context()
    click.echo(ctx.get_help())


@click.group(name="truss", invoke_without_command=True)  # type: ignore
@click.pass_context
@click.version_option(truss.version())
@log_level_option
def truss_cli(ctx) -> None:
    """truss: The simplest way to serve models in production"""
    if not ctx.invoked_subcommand:
        click.echo(ctx.get_help())


@click.group()
def container():
    """Subcommands for truss container"""


@click.group()
def image():
    """Subcommands for truss image"""


class ChainsGroup(click.Group):
    def invoke(self, ctx: click.Context) -> None:
        # This import raises error messages if pydantic v2 or python older than 3.9
        # are installed.
        import truss_chains  # noqa: F401

        super().invoke(ctx)


@click.group(cls=ChainsGroup)
def chains():
    """Subcommands for truss chains"""


@truss_cli.command()
@click.argument("target_directory", required=True)
@click.option(
    "-b",
    "--backend",
    show_default=True,
    default=ModelServer.TrussServer.value,
    type=click.Choice([server.value for server in ModelServer]),
)
@log_level_option
@error_handling
def init(target_directory, backend) -> None:
    """Create a new truss.

    TARGET_DIRECTORY: A Truss is created in this directory
    """
    if os.path.isdir(target_directory):
        raise click.ClickException(
            f"Error: Directory `{target_directory}` already exists "
            "and cannot be overwritten."
        )
    tr_path = Path(target_directory)
    build_config = Build(model_server=ModelServer[backend])
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
@log_level_option
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
@click.option("--tag", help="Docker image tag")
@log_level_option
@error_handling
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
@log_level_option
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
@log_level_option
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

    service = remote_provider.get_service(model_identifier=ModelName(model_name))
    rich.print(f"🪵  View logs for your deployment at {_format_link(service.logs_url)}")
    remote_provider.sync_truss_to_dev_version_by_name(model_name, target_directory)


def _create_chains_table(service) -> Tuple[rich.table.Table, List[str]]:
    """Creates a status table e.g.

                                                     Deployments
    ╭──────────────────────┬──────────────────────┬────────────────────────────╮
    │ Status               │ Chainlet             │                   Logs URL │
    ├──────────────────────┼──────────────────────┼────────────────────────────┤
    │  👾 DEPLOYING        │ SplitText            │ https://app.baseten.co/... │
    │  👾 DEPLOYING        │ GenerateData         │ https://app.baseten.co/... │
    │  👾 DEPLOYING        │ MistralLLM           │ https://app.baseten.co/... │
    │  👾 DEPLOYING        │ TextToNum            │ https://app.baseten.co/... │
    │  👾 DEPLOYING        │ Chain                │ https://app.baseten.co/... │
    ╰──────────────────────┴──────────────────────┴────────────────────────────╯
    """
    title = (
        f"⛓️   {service.name} - Chain  ⛓️\n\n "
        f"🌐 Status page: {_format_link(service.status_page_url)}"
    )
    table = rich.table.Table(
        show_header=True,
        header_style="bold yellow",
        title=title,
        box=rich.table.box.ROUNDED,
        border_style="blue",
    )
    table.add_column("Status", style="dim", min_width=20)
    table.add_column("Chainlet", min_width=20)
    table.add_column("Logs URL")
    statuses = []
    # After reversing, the first one is the entrypoint (per order in service).
    for i, chainlet in enumerate(reversed(service.get_info())):
        displayable_status = get_displayable_status(chainlet.status)
        if displayable_status == ACTIVE_STATUS:
            spinner_name = "active"
        elif displayable_status in DEPLOYING_STATUSES:
            if displayable_status == "BUILDING":
                spinner_name = "building"
            elif displayable_status == "LOADING":
                spinner_name = "loading"
            else:
                spinner_name = "deploying"
        else:
            spinner_name = "failed"
        spinner = rich.spinner.Spinner(spinner_name, text=displayable_status)
        if chainlet.is_entrypoint:
            display_name = f"{chainlet.name} (entrypoint)"
        else:
            display_name = f"{chainlet.name} (internal)"

        table.add_row(spinner, display_name, _format_link(chainlet.logs_url))
        if chainlet.is_entrypoint:  # Add section divider after entrypoint.
            table.add_section()
        statuses.append(displayable_status)
    return table, statuses


@chains.command()  # type: ignore
@click.argument("source", type=Path, required=True)
@click.argument("entrypoint", type=str, required=False)
@click.option(
    "--name",
    type=str,
    required=False,
    help="Name of the chain to be deployed, if not given, the entrypoint name is used.",
)
@click.option(
    "--publish",
    type=bool,
    default=True,
    is_flag=True,
    help="Create chainlets as published deployments.",
)
@click.option(
    "--promote",
    type=bool,
    default=False,
    is_flag=True,
    help="Replace production chainlets with newly deployed chainlets.",
)
@click.option(
    "--wait",
    type=bool,
    default=True,
    is_flag=True,
    help="Wait until all chainlets are ready (or deployment failed).",
)
@click.option(
    "--dryrun",
    type=bool,
    default=False,
    is_flag=True,
    help="Produces only generated files, but doesn't deploy anything.",
)
@click.option(
    "--remote",
    type=str,
    required=False,
    help="Name of the remote in .trussrc to push to.",
)
@log_level_option
@error_handling
def deploy(
    source: Path,
    entrypoint: Optional[str],
    name: Optional[str],
    publish: bool,
    promote: bool,
    wait: bool,
    dryrun: bool,
    remote: Optional[str],
) -> None:
    """
    Deploys a chain remotely.

    SOURCE: Path to a python file that contains the entrypoint chainlet.

    ENTRYPOINT: Class name of the entrypoint chainlet in source file. May be omitted
    if a chainlet definition in SOURCE is tagged with `@chains.mark_entrypoint`.
    """
    # These imports are delayed, to handle pydantic v1 envs gracefully.
    from truss_chains import definitions as chains_def
    from truss_chains import deploy as chains_deploy
    from truss_chains import framework

    with framework.import_target(source, entrypoint) as entrypoint_cls:
        chain_name = name or entrypoint_cls.__name__
        options = chains_def.DeploymentOptionsBaseten.create(
            chain_name=chain_name,
            promote=promote,
            publish=publish,
            only_generate_trusses=dryrun,
            remote=remote,
        )
        service = chains_deploy.deploy_remotely(entrypoint_cls, options)

    console.print("\n")
    if dryrun:
        return

    run_help_msg = (
        f"curl -X POST '{service.run_url}' \\\n"
        '    -H "Authorization: Api-Key $BASETEN_API_KEY" \\\n'
        "    -d '<JSON_INPUT>'"
    )

    table, statuses = _create_chains_table(service)
    status_check_wait_sec = 2
    if wait:
        num_services = len(statuses)
        success = False
        num_failed = 0
        with rich.live.Live(table, console=console, refresh_per_second=4) as live:
            while True:
                table, statuses = _create_chains_table(service)
                live.update(table)
                num_active = sum(s == ACTIVE_STATUS for s in statuses)
                num_deploying = sum(s in DEPLOYING_STATUSES for s in statuses)
                if num_active == num_services:
                    success = True
                    break
                elif num_failed := num_services - num_active - num_deploying:
                    break
                time.sleep(status_check_wait_sec)
        # Print must be outside `Live` context.
        if success:
            console.print("Deployment succeeded.", style="bold green")
            console.print(f"You can run the chain with:\n{run_help_msg}")
        else:
            console.print(f"Deployment failed ({num_failed} failures).", style="red")
    else:
        console.print(table)
        console.print(
            "Once all chainlets are deployed, "
            f"you can run the chain with:\n\n{run_help_msg}"
        )


@chains.command(name="init")  # type: ignore
@click.argument("directory", type=Path, required=False)
@log_level_option
@error_handling
def chains_init(
    directory: Optional[Path],
) -> None:
    """
    Initializes a chains project directory.

    DIRECTORY: A name of new or existing directory to create the chain in,
      it must be empty. If not specified, the current directory is used.

    """
    if not directory:
        directory = Path.cwd()
    if directory.exists():
        if not directory.is_dir():
            raise ValueError(f"The path {directory} must be a directory.")
        if any(directory.iterdir()):
            raise ValueError(f"Directory {directory} must be empty.")
    else:
        directory.mkdir()

    filename = inquirer.text(
        qmark="",
        message="Enter the python file name for the chain.",
        default="my_chain.py",
    ).execute()
    filepath = directory / str(filename).strip()
    rich.print(f"Creating and populating {filepath}...\n")
    source_code = _load_example_chainlet_code()
    filepath.write_text(source_code)
    rich.print(
        "Next steps:\n",
        f"💻 Run [bold green]`python {filepath}`[/bold green] for local debug "
        "execution.\n"
        f"🚢 Run [bold green]`truss chains deploy {filepath}`[/bold green] "
        "to deploy the chain to Baseten.\n",
    )


def _load_example_chainlet_code() -> str:
    try:
        from truss_chains import example_chainlet
    # if the example is faulty, a validation error would be raised
    except Exception as e:
        raise Exception("Failed to load starter code. Please notify support.") from e

    source = Path(example_chainlet.__file__).read_text()
    return source


def _extract_and_validate_model_identifier(
    target_directory: str,
    model_id: Optional[str],
    model_version_id: Optional[str],
    published: Optional[bool],
) -> ModelIdentifier:
    if published and (model_id or model_version_id):
        raise click.UsageError(
            "Cannot use --published with --model or --model-deployment."
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
    help=(
        "[DEPRECATED] Use --model-deployment instead, this will be  "
        "removed in future release. ID of model deployment"
    ),
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
@log_level_option
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
            "[DEPRECATED] --model-version is deprecated, "
            "use --model-deployment instead.",
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
        "deployment exists, promote the truss to production "
        "after deploy completes."
    ),
)
@click.option(
    "--promote",
    type=bool,
    is_flag=True,
    required=False,
    default=False,
    help=(
        "Push the truss as a published deployment. Even if a production "
        "deployment exists, promote the truss to production "
        "after deploy completes."
    ),
)
@click.option(
    "--preserve-previous-production-deployment",
    type=bool,
    is_flag=True,
    required=False,
    default=False,
    help=(
        "Preserve the previous production deployment's autoscaling setting. When "
        "not specified, the previous production deployment will be updated to allow "
        "it to scale to zero. Can only be use in combination with --promote option."
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
    "--deployment-name",
    type=str,
    required=False,
    help=(
        "Name of the deployment created by the push. Can only be "
        "used in combination with `--publish` or `--promote`."
    ),
)
@click.option(
    "--wait/--no-wait",
    type=bool,
    is_flag=True,
    required=False,
    default=False,
    help="Wait for the deployment to complete before returning.",
)
@click.option(
    "--timeout-seconds",
    type=int,
    required=False,
    help=(
        "Maximum time to wait for deployment to complete in seconds. Without "
        "specifying, the command will not complete until the deployment is complete."
    ),
)
@log_level_option
@error_handling
def push(
    target_directory: str,
    remote: str,
    model_name: str,
    publish: bool = False,
    trusted: bool = False,
    promote: bool = False,
    preserve_previous_production_deployment: bool = False,
    deployment_name: Optional[str] = None,
    wait: bool = False,
    timeout_seconds: Optional[int] = None,
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

    # Log a warning if using secrets without --trusted.
    # TODO(helen): this could be moved to a separate function that includes more
    #  config checks.
    if tr.spec.config.secrets and not trusted:
        not_trusted_text = (
            "Warning: your Truss has secrets but was not pushed with --trusted. "
            "Please push with --trusted to grant access to secrets."
        )
        console.print(not_trusted_text, style="red")

    # trt-llm engine builder checks
    if not check_secrets_for_trt_llm_builder(tr):
        missing_token_text = (
            "`hf_access_token` must be provided in secrets to build a gated model. "
            "Please see https://docs.baseten.co/deploy/guides/private-model for configuration instructions."
        )
        console.print(missing_token_text, style="red")
        sys.exit(1)
    if not check_and_update_memory_for_trt_llm_builder(tr):
        console.print(
            f"Automatically increasing memory for trt-llm builder to {TRTLLM_MIN_MEMORY_REQUEST_GI}Gi."
        )

    # TODO(Abu): This needs to be refactored to be more generic
    service = remote_provider.push(
        tr,
        model_name=model_name,
        publish=publish,
        trusted=trusted,
        promote=promote,
        preserve_previous_prod_deployment=preserve_previous_production_deployment,
        deployment_name=deployment_name,
    )  # type: ignore

    click.echo(f"✨ Model {model_name} was successfully pushed ✨")

    if service.is_draft:
        draft_model_text = """
|---------------------------------------------------------------------------------------|
| Your model is deploying as a development model. Development models allow you to  |
| iterate quickly during the deployment process.                                        |
|                                                                                       |
| When you are ready to publish your deployed model as a new deployment,                |
| pass `--publish` to the `truss push` command. To monitor changes to your model and    |
| rapidly iterate, run the `truss watch` command.                                       |
|                                                                                       |
|---------------------------------------------------------------------------------------|
"""

        click.echo(draft_model_text)

    if promote:
        promotion_text = (
            "Your Truss has been deployed as a production model. After it successfully "
            "deploys, it will become the next production deployment of your model."
        )
        console.print(promotion_text, style="green")

    rich.print(f"🪵  View logs for your deployment at {_format_link(service.logs_url)}")
    if wait:
        start_time = time.time()
        with console.status("[bold green]Deploying...") as status:
            try:
                # Poll for the deployment status until we have reached. Either ACTIVE,
                # or a non-deploying status (in which case the deployment has failed).
                for deployment_status in service.poll_deployment_status():
                    if (
                        timeout_seconds is not None
                        and time.time() - start_time > timeout_seconds
                    ):
                        console.print("Deployment timed out.", style="red")
                        sys.exit(1)

                    status.update(
                        f"[bold green]Deploying...Current Status: {deployment_status}"
                    )

                    if deployment_status == ACTIVE_STATUS:
                        console.print("Deployment succeeded.", style="bold green")
                        return

                    if deployment_status not in DEPLOYING_STATUSES:
                        console.print(
                            f"Deployment failed with status {deployment_status}.",
                            style="red",
                        )
                        sys.exit(1)

            except RemoteNetworkError:
                console.print("Deployment failed: Could not reach remote.", style="red")
                sys.exit(1)


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
    """Kills all truss containers that are not manually persisted."""
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
truss_cli.add_command(chains)

if __name__ == "__main__":
    truss_cli()
