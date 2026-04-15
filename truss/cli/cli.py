import inspect
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Optional, cast

import rich.table
import rich_click as click
from rich import console as rich_console
from rich import progress

import truss
from truss.base.constants import (
    PRODUCTION_ENVIRONMENT_NAME,
    TRTLLM_MIN_MEMORY_REQUEST_GI,
)
from truss.base.trt_llm_config import TrussTRTLLMQuantizationType
from truss.base.truss_config import Build, ModelServer, TransportKind
from truss.cli import remote_cli
from truss.cli.logs import utils as cli_log_utils
from truss.cli.logs.model_log_watcher import ModelDeploymentLogWatcher
from truss.cli.resolvers.model_team_resolver import (
    resolve_model_for_watch,
    resolve_model_team_name,
)
from truss.cli.utils import common, self_upgrade
from truss.cli.utils.output import console, error_console, json_command
from truss.remote.baseten.core import (
    ACTIVE_STATUS,
    DEPLOYING_STATUSES,
    ModelId,
    ModelIdentifier,
    ModelName,
    ModelVersionId,
    get_dev_version_from_versions,
)
from truss.remote.baseten.remote import BasetenRemote
from truss.remote.baseten.service import BasetenService, URLConfig
from truss.remote.remote_factory import USER_TRUSSRC_PATH, RemoteFactory
from truss.trt_llm.config_checks import (
    has_no_tags_trt_llm_builder,
    memory_updated_for_trt_llm_builder,
    uses_trt_llm_builder,
)
from truss.truss_handle.build import cleanup as _cleanup
from truss.truss_handle.build import init_directory as _init
from truss.truss_handle.build import load
from truss.util import docker, user_config

click.rich_click.COMMAND_GROUPS = {
    "truss": [
        {
            "name": "Main usage",
            "commands": ["init", "push", "watch", "predict", "model_logs"],
            "table_styles": {"row_styles": ["green"]},  # type: ignore
        },
        {
            "name": "Advanced Usage",
            "commands": ["image", "container", "cleanup"],
            "table_styles": {"row_styles": ["yellow"]},  # type: ignore
        },
        {
            "name": "Chains",
            "commands": ["chains"],
            "table_styles": {"row_styles": ["red"]},  # type: ignore
        },
        {
            "name": "Train",
            "commands": ["train"],
            "table_styles": {"row_styles": ["magenta"]},  # type: ignore
        },
    ]
}


def _get_truss_from_directory(
    target_directory: Optional[str] = None, config: Optional[str] = None
):
    """Gets Truss from directory. If none, use the current directory"""
    if target_directory is None:
        target_directory = os.getcwd()
    config_path = Path(config) if config else None
    if not os.path.isfile(target_directory):
        return load(target_directory, config_path=config_path)
    # These imports are delayed, to handle pydantic v1 envs gracefully.
    from truss_chains.deployment import code_gen

    truss_dir = code_gen.gen_truss_model_from_source(Path(target_directory))
    return load(truss_dir, config_path=config_path)


def _start_tail(
    remote: BasetenRemote, model_id: str, version_id: str, in_background: bool
) -> None:
    log_watcher = ModelDeploymentLogWatcher(remote.api, model_id, version_id)

    def _tail_logs():
        try:
            for log in log_watcher.watch(show_spinner=not in_background):
                cli_log_utils.output_log(log)
        except Exception as exc:
            error_console.print(
                f"[red]Log tailing stopped due to an error:[/red] {exc}"
            )
            raise

    if in_background:
        thread = threading.Thread(target=_tail_logs, daemon=True)
        thread.start()
    else:
        _tail_logs()


def _start_watch_mode(
    target_directory: str,
    model_name: str,
    remote_provider: BasetenRemote,
    resolved_model: dict,
    resolved_versions: list,
    console: "rich_console.Console",
    error_console: "rich_console.Console",
    hot_reload: bool = False,
) -> None:
    if not os.path.isfile(target_directory):
        remote_provider.sync_truss_to_dev_version_with_model(
            resolved_model,
            resolved_versions,
            target_directory,
            console,
            error_console,
            hot_reload=hot_reload,
        )
    else:
        # These imports are delayed, to handle pydantic v1 envs gracefully.
        from truss_chains.deployment import deployment_client

        deployment_client.watch_model(
            source=Path(target_directory),
            model_name=model_name,
            remote_provider=remote_provider,
            console=console,
            error_console=error_console,
        )


### Top-level & utility commands. ######################################################


@click.group(
    name="truss",
    invoke_without_command=True,
    context_settings=dict(help_option_names=["-h", "--help"]),
)  # type: ignore
@click.pass_context
@click.version_option(truss.__version__)
@common.common_options(add_middleware=False)
def truss_cli(ctx) -> None:
    """truss: The simplest way to serve models in production"""
    # Click "stacks" the root command and group/subcommands, to avoid running the
    # middleware twice, we don't add it via decorator to the root command, but instead
    # selective run it here inline.
    if not ctx.invoked_subcommand:
        common.set_logging_level()
        common.upgrade_dialogue()
        click.echo(ctx.get_help())


@truss_cli.command()
@click.option("--api-key", type=str, required=False, help="API key for authentication.")
@common.common_options()
def login(api_key: Optional[str]):
    from truss.api import login

    if not api_key:
        remote_config = remote_cli.inquire_remote_config()
        RemoteFactory.update_remote_config(remote_config)
    else:
        login(api_key)


@truss_cli.command()
@click.argument("version", required=False)
@common.common_options()
@click.pass_context
def upgrade(ctx: click.Context, version: Optional[str]) -> None:
    """Upgrade truss to the latest (or specified) version."""
    interactive = not ctx.obj.get("non_interactive", False)
    self_upgrade.run_upgrade(version, interactive=interactive)


def _create_oidc_table(oidc_info) -> rich.table.Table:
    """Creates an OIDC information table."""
    table = rich.table.Table(
        show_header=False,
        title="OIDC Configuration for Workload Identity",
        box=rich.table.box.ROUNDED,
        border_style="blue",
    )
    table.add_column(style="cyan", min_width=20)
    table.add_column(min_width=40)
    table.add_row("Org ID", oidc_info.org_id)

    if oidc_info.teams:
        teams_display = ", ".join(
            f"{team.id} ({team.name})" for team in oidc_info.teams
        )
        table.add_row("Teams", teams_display)
    else:
        table.add_row("Teams", "[ ]")

    table.add_row("Issuer", oidc_info.issuer)
    table.add_row("Audience", oidc_info.audience)
    table.add_row("Workload Type Options", ", ".join(oidc_info.workload_types))

    table.add_section()
    table.add_row(
        "Subject Claim Format",
        "v=1:org=<org_id>:team=<team_id>:model=<model_id>:"
        "deployment=<deployment_id>:environment=<environment>:type=<workload_type>",
    )

    return table


@truss_cli.command()
@click.option(
    "--remote",
    type=str,
    required=False,
    help="Name of the remote in .trussrc to check whoami.",
)
@click.option(
    "--show-oidc",
    is_flag=True,
    default=False,
    help="Show OIDC configuration for workload identity.",
)
@common.common_options()
def whoami(remote: Optional[str], show_oidc: bool):
    """
    Shows user information and exit.
    """
    from truss.api import whoami

    if not remote:
        remote = remote_cli.inquire_remote_name()

    user = whoami(remote)

    console.print(f"{user.workspace_name}\\{user.user_email}")

    if show_oidc:
        remote_provider = cast(BasetenRemote, RemoteFactory.create(remote=remote))
        oidc_info = remote_provider.get_oidc_info()

        console.print()
        table = _create_oidc_table(oidc_info)
        console.print(table)
        console.print(
            f"Learn more: {common.format_link('https://docs.baseten.co/organization/oidc')}"
        )


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


@truss_cli.command()
@common.common_options()
def cleanup() -> None:
    """
    Clean up truss data.

    Truss creates temporary directories for various operations
    such as for building docker images. This command clears
    that data to free up disk space.
    """
    _cleanup()


### Truss (model) commands. ############################################################


@truss_cli.command()
@click.argument("target_directory", required=True)
@click.option(
    "-b",
    "--backend",
    show_default=True,
    default=ModelServer.TrussServer.value,
    type=click.Choice([server.value for server in ModelServer]),
)
@click.option("-n", "--name", type=click.STRING)
@click.option(
    "--python-config/--no-python-config",
    default=False,
    help="Uses the code first tooling to build models.",
)
@common.common_options()
def init(target_directory, backend, name, python_config) -> None:
    """Create a new truss.

    TARGET_DIRECTORY: A Truss is created in this directory
    """
    if os.path.isdir(target_directory):
        raise click.ClickException(
            f"Error: Directory '{target_directory}' already exists "
            "and cannot be overwritten."
        )
    tr_path = Path(target_directory)
    build_config = Build(model_server=ModelServer[backend])
    if name:
        model_name = name
    else:
        model_name = remote_cli.inquire_model_name()
    _init(
        target_directory=target_directory,
        build_config=build_config,
        model_name=model_name,
        python_config=python_config,
    )
    click.echo(f"Truss {model_name} was created in {tr_path.absolute()}")


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
    help="Name of the remote in .trussrc to push to.",
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
@click.option("--model", type=str, required=False, help="ID of model to call")
@common.log_level_option
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
        remote = remote_cli.inquire_remote_name()

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
        console.print(
            f"Calling predict on {'[cyan]development[/cyan] ' if service.is_draft else ''}"
            f"deployment ID {service.model_version_id}..."
        )

    result = service.predict(request_data)
    if inspect.isgenerator(result):
        for chunk in result:
            click.echo(chunk, nl=False)
        return
    console.print_json(data=result)


@truss_cli.command()
@click.argument("script", required=True)
@click.argument("target_directory", required=False, default=os.getcwd())
def run_python(script, target_directory):
    if not Path(script).exists():
        raise click.BadParameter(
            f"File {script} does not exist. Please provide a valid file."
        )

    if not Path(target_directory).exists():
        raise click.BadParameter(f"Directory {target_directory} does not exist.")

    if not (Path(target_directory) / "config.yaml").exists():
        raise click.BadParameter(
            f"Directory {target_directory} does not contain a valid Truss."
        )

    tr = _get_truss_from_directory(target_directory=target_directory)
    container_ = tr.run_python_script(Path(script))
    for output in container_.logs():
        output_type = output[0]
        output_content = output[1]

        options = {}

        if output_type == "stderr":
            options["fg"] = "red"

        click.secho(output_content.decode("utf-8", "replace"), nl=False, **options)
    exit_code = container.wait()
    sys.exit(exit_code)


@truss_cli.command()
@click.argument("target_directory", required=False, default=os.getcwd())
@click.option(
    "--config",
    type=click.Path(exists=True),
    required=False,
    help="Path to a custom config file (default: config.yaml in truss directory)",
)
@click.option(
    "--remote",
    type=str,
    required=False,
    help="Name of the remote in .trussrc to push to.",
)
@click.option(
    "--model-name",
    type=str,
    required=False,
    help="Temporarily override the name of the model",
)
@click.option(
    "--publish",
    is_flag=True,
    required=False,
    default=False,
    help=(
        "[DEPRECATED] Published deployments are now the default."
        "Push the truss as a published deployment. If no production "
        "deployment exists, promote the truss to production "
        "after deploy completes."
    ),
)
@click.option(
    "--promote",
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
    "--environment",
    type=str,
    required=False,
    help=(
        "Push the truss as a published deployment to the specified environment."
        "If specified, publish is implied and the supplied value of --promote will be ignored."
    ),
)
@click.option(
    "--preserve-previous-production-deployment",
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
    is_flag=True,
    required=False,
    default=None,
    help="[DEPRECATED] All models are trusted by default.",
)
@click.option(
    "--disable-truss-download",
    is_flag=True,
    required=False,
    default=False,
    help="Disable downloading the truss directory from the UI.",
)
@click.option(
    "--deployment-name",
    type=str,
    required=False,
    help=(
        "Name of the deployment created by the push. Cannot be "
        "used with --watch (development deployments)."
    ),
)
@click.option(
    "--wait/--no-wait",
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
@click.option(
    "--include-git-info",
    is_flag=True,
    required=False,
    default=False,
    help=common.INCLUDE_GIT_INFO_DOC,
)
@click.option("--tail", is_flag=True)
@click.option(
    "--preserve-env-instance-type/--no-preserve-env-instance-type",
    is_flag=True,
    required=False,
    default=None,
    help=(
        "When pushing a truss to an environment, whether to use the resources specified "
        "in the truss config to resolve the instance type or preserve the instance type "
        "configured in the specified environment. It will be ignored if --environment is not specified. "
        "Default is --preserve-env-instance-type."
    ),
)
@click.option(
    "--deploy-timeout-minutes",
    type=int,
    required=False,
    help="Timeout in minutes for the deploy operation.",
)
@click.option(
    "--team",
    "provided_team_name",
    type=str,
    required=False,
    help="Team name for the model",
)
@click.option(
    "--labels",
    type=str,
    required=False,
    help="JSON string of labels as key-value pairs.",
)
@click.option(
    "--watch",
    "watch_after_push",
    is_flag=True,
    required=False,
    default=False,
    help=(
        "Deploy as a development model and watch for changes. "
        "Waits for deployment to complete, then starts watching for code changes "
        "to apply live patches. Cannot be used with --promote or --environment."
    ),
)
@click.option(
    "--watch-hot-reload",
    "watch_hot_reload",
    is_flag=True,
    required=False,
    default=False,
    help=(
        "Enable hot-reload for model code changes during watch mode. "
        "Swaps model class in-process without restarting, preserving state. "
        "Requires --watch."
    ),
)
@click.option(
    "--no-cache",
    "no_cache",
    is_flag=True,
    required=False,
    default=False,
    help="Force a full rebuild without using cached layers.",
)
@click.option(
    "--watch-no-sleep",
    is_flag=True,
    required=False,
    default=False,
    help="Keep the development model warm by preventing scale-to-zero while watching. Requires --watch.",
)
@click.option(
    "--output",
    "output_format",
    type=click.Choice(["text", "json"]),
    default="text",
    help=(
        "Output format. 'json' emits structured JSON to stdout and "
        "all other output (progress, logs) to stderr."
    ),
)
@common.common_options()
@json_command
def push(
    target_directory: str,
    config: Optional[str],
    remote: str,
    model_name: str,
    publish: bool = False,
    trusted: Optional[bool] = None,
    disable_truss_download: bool = False,
    promote: bool = False,
    preserve_previous_production_deployment: bool = False,
    deployment_name: Optional[str] = None,
    wait: bool = False,
    timeout_seconds: Optional[int] = None,
    environment: Optional[str] = None,
    include_git_info: bool = False,
    tail: bool = False,
    preserve_env_instance_type: bool = True,
    deploy_timeout_minutes: Optional[int] = None,
    provided_team_name: Optional[str] = None,
    labels: Optional[str] = None,
    watch_after_push: bool = False,
    watch_hot_reload: bool = False,
    no_cache: bool = False,
    watch_no_sleep: bool = False,
    output_format: str = "text",
) -> None:
    """
    Pushes a truss to a TrussRemote.

    TARGET_DIRECTORY: A Truss directory. If none, use current directory.

    """

    if publish:
        console.print(
            "[DEPRECATED] The --publish flag is deprecated. Published deployments are now the default.",
            style="yellow",
        )

    if watch_no_sleep and not watch_after_push:
        raise click.UsageError(
            "Cannot use --watch-no-sleep without --watch. --watch-no-sleep prevents scale-to-zero during watch mode."
        )

    # Handle --watch flag: deploys as development and then watches
    if watch_after_push:
        if publish:
            raise click.UsageError(
                "Cannot use --watch with --publish. Watch mode requires a development deployment."
            )
        if promote:
            raise click.UsageError(
                "Cannot use --watch with --promote. Watch mode runs a development deployment."
            )
        if environment:
            raise click.UsageError(
                "Cannot use --watch with --environment. Watch mode runs a development deployment."
            )
        # Development deployment for watch mode
        publish = False
        wait = True
    else:
        if watch_hot_reload:
            raise click.UsageError("--watch-hot-reload requires --watch.")
        # Default is now published deployment
        publish = True
        console.print(
            "Deploying as a published deployment. Use --watch for a development deployment.",
            style="green",
        )

    tr = _get_truss_from_directory(target_directory=target_directory, config=config)

    if tr.spec.config.resources.instance_type:
        console.print(
            "Field 'instance_type' specified - ignoring 'cpu', 'memory', 'accelerator', and 'use_gpu' fields.",
            style="yellow",
        )

    if (
        tr.spec.config.runtime.transport.kind == TransportKind.GRPC
        and not publish
        and not promote
    ):
        raise click.UsageError(
            "Truss with gRPC transport cannot be used as a development deployment. Remove --watch to deploy as a published model."
        )

    if not remote:
        remote = remote_cli.inquire_remote_name()

    if not include_git_info:
        include_git_info = user_config.settings.include_git_info

    remote_provider = RemoteFactory.create(remote=remote)
    if output_format == "json" and isinstance(remote_provider, BasetenRemote):
        remote_provider.api.suppress_error_print = True

    # model_name from CLI flag (explicit), or fall back to config
    cli_model_name = model_name  # what the user inputted for `--model-name`
    model_name = model_name or tr.spec.config.model_name
    if not model_name:
        model_name = remote_cli.inquire_model_name()

    # Only persist back to config.yaml if it wasn't provided on the CLI
    if not cli_model_name and model_name != tr.spec.config.model_name:
        tr.spec.config.model_name = model_name
        tr.spec.config.write_to_yaml_file(tr.spec.config_path, verbose=False)

    # Resolve team_id if BasetenRemote
    team_id = None
    if isinstance(remote_provider, BasetenRemote):
        existing_teams = remote_provider.api.get_teams()
        team_name, team_id = resolve_model_team_name(
            remote_provider=remote_provider,
            provided_team_name=provided_team_name,
            existing_model_name=model_name,
            existing_teams=existing_teams,
            remote_name=remote,
        )

    if promote and environment:
        raise click.UsageError(
            "'promote' flag and 'environment' flag cannot both be specified."
        )
    if promote and not environment:
        environment = PRODUCTION_ENVIRONMENT_NAME

    if preserve_env_instance_type is not None and not environment:
        preserve_env_warning = "'preserve-env-instance-type' flag specified without the 'environment' parameter. Ignoring the value of `preserve-env-instance-type`"
        console.print(preserve_env_warning, style="yellow")
    if preserve_env_instance_type is None:
        # If the flag is not specified, we set it to True by default. We handle the default here instead of in click.options
        # to only print the warning above when the flag was specified by the user.
        preserve_env_instance_type = True

    if environment:
        if preserve_env_instance_type:
            preserve_env_info = f"'preserve-env-instance-type' used. Resources from the config will be ignored and the current instance type of the '{environment}' environment will be used."
            console.print(preserve_env_info)
        else:
            preserve_env_info = f"'no-preserve-env-instance-type' used. Instance type will be derived from the config and updated in the '{environment}' environment."
            console.print(preserve_env_info)

    # Log a warning if using --trusted.
    if trusted is not None:
        trusted_deprecation_notice = "[DEPRECATED] '--trusted' option is deprecated and no longer needed. All models are trusted by default."
        console.print(trusted_deprecation_notice, style="yellow")

    if no_cache:
        tr.spec.config.build.no_cache = True

    # Parse labels from CLI option
    labels_dict: Optional[dict] = None
    if labels:
        try:
            parsed_labels = json.loads(labels)
            if not isinstance(parsed_labels, dict):
                raise click.UsageError("--labels must be a JSON object.")
            labels_dict = parsed_labels
        except json.JSONDecodeError as e:
            raise click.UsageError(f"Invalid JSON in --labels: {e}")

    # trt-llm engine builder checks
    if uses_trt_llm_builder(tr):
        if not publish:
            live_reload_disabled_text = "Development mode is currently not supported for trusses using TRT-LLM build flow. Remove --watch to deploy as a published model."
            console.print(live_reload_disabled_text, style="red")
            sys.exit(1)

        if memory_updated_for_trt_llm_builder(tr):
            console.print(
                f"Automatically increasing memory for trt-llm builder to {TRTLLM_MIN_MEMORY_REQUEST_GI}Gi."
            )
        message_oai, raised_message_oai = has_no_tags_trt_llm_builder(tr)
        if message_oai:
            console.print(message_oai, style="yellow")
            if raised_message_oai:
                console.print(message_oai, style="red")
                sys.exit(1)

        trt_llm_build_config = tr.spec.config.trt_llm.build
        if (
            trt_llm_build_config.quantization_type
            in [TrussTRTLLMQuantizationType.FP8, TrussTRTLLMQuantizationType.FP8_KV]
            and not trt_llm_build_config.num_builder_gpus
        ):
            fp8_and_num_builder_gpus_text = (
                "Warning: build specifies FP8 quantization but does not explicitly specify number of build GPUs. "
                "GPU memory required at build time may be significantly more than that required at inference time due to FP8 quantization, which can result in OOM failures during the engine build phase."
                "'num_builder_gpus' can be used to specify the number of GPUs to use at build time."
            )
            console.print(fp8_and_num_builder_gpus_text, style="yellow")

    source = Path(target_directory)
    working_dir = source.parent if source.is_file() else source.resolve()

    service = remote_provider.push(
        truss_handle=tr,
        model_name=model_name,
        working_dir=working_dir,
        publish=publish,
        promote=promote,
        preserve_previous_prod_deployment=preserve_previous_production_deployment,
        deployment_name=deployment_name,
        environment=environment,
        disable_truss_download=disable_truss_download,
        progress_bar=progress.Progress,
        include_git_info=include_git_info,
        preserve_env_instance_type=preserve_env_instance_type,
        deploy_timeout_minutes=deploy_timeout_minutes,
        team_id=team_id,
        labels=labels_dict,
    )

    console.print(f"✨ Model {model_name} was successfully pushed ✨")

    if service.is_draft:
        draft_model_text = """
|---------------------------------------------------------------------------------------|
| Your model is deploying as a development model. Development models allow you to       |
| iterate quickly during the deployment process.                                        |
|                                                                                       |
| When you are ready to publish your deployed model as a new deployment,                |
| run 'truss push' without --watch.                                                     |
|                                                                                       |
|---------------------------------------------------------------------------------------|
"""

        console.print(draft_model_text)

    if environment:
        promotion_text = (
            f"Your Truss has been deployed into the {environment} environment. After it successfully "
            f"deploys, it will become the next {environment} deployment of your model."
        )
        console.print(promotion_text, style="green")

    console.print(
        f"🪵  View logs for your deployment at {common.format_link(service.logs_url)}"
    )

    if tail and isinstance(service, BasetenService):
        # When combined with --wait/--watch, tail runs in background so the
        # wait polling loop below can proceed on the main thread.
        _start_tail(
            cast(BasetenRemote, remote_provider),
            service.model_id,
            service.model_version_id,
            in_background=wait,
        )

    last_deployment = None
    if wait:
        start_time = time.time()
        with console.status("[bold green]Deploying...") as status:
            for deployment in service.poll_deployment():
                last_deployment = deployment
                deployment_status = deployment["status"]
                if (
                    timeout_seconds is not None
                    and time.time() - start_time > timeout_seconds
                ):
                    raise TimeoutError("Deployment timed out.")

                status.update(
                    f"[bold green]Deploying...Current Status: {deployment_status}"
                )

                if deployment_status == ACTIVE_STATUS:
                    console.print("Deployment succeeded.", style="bold green")
                    break

                # For --watch (dev deployments), enter watch mode early
                # once past BUILDING, so user can iterate on code
                if watch_after_push and deployment_status in ("LOADING_MODEL"):
                    console.print(
                        f"Deployment status: {deployment_status}. "
                        "Entering watch mode early for faster iteration...",
                        style="bold blue",
                    )
                    break

                if deployment_status not in DEPLOYING_STATUSES:
                    exc = RuntimeError(
                        f"Deployment failed with status {deployment_status}."
                    )
                    setattr(exc, "json_data", {"deployment": deployment})
                    raise exc

        # If --watch was used, start watching after deploy success
        if watch_after_push:
            if not isinstance(remote_provider, BasetenRemote):
                raise click.UsageError(
                    f"Watch mode is not supported for remote provider type: {type(remote_provider).__name__}"
                )
            bt_remote = cast(BasetenRemote, remote_provider)
            console.print("Starting watch mode...", style="bold blue")
            resolved_model, versions = resolve_model_for_watch(
                bt_remote, model_name, provided_team_name=team_name
            )
            if watch_no_sleep:
                model_hostname = resolved_model["hostname"]
                api_key = bt_remote._auth_service.authenticate().value
                common.start_keepalive(model_hostname, api_key)
            _start_watch_mode(
                target_directory=target_directory,
                model_name=model_name,
                remote_provider=bt_remote,
                resolved_model=resolved_model,
                resolved_versions=versions,
                console=console,
                error_console=error_console,
                hot_reload=watch_hot_reload,
            )

    if output_format == "json" and isinstance(service, BasetenService):
        result: dict = {
            "model_id": service.model_id,
            "model_version_id": service.model_version_id,
            "predict_url": service.predict_url,
            "logs_url": service.logs_url,
            "is_draft": service.is_draft,
        }
        if last_deployment is not None:
            result["deployment"] = last_deployment
        print(json.dumps(result, indent=2), file=sys.stdout)


@truss_cli.command()
@click.option(
    "--remote", type=str, required=False, help="Name of the remote in .trussrc."
)
@click.option("--model-id", type=str, required=True)
@click.option("--deployment-id", type=str, required=True)
@click.option("--tail", is_flag=True, help="Tail for ongoing logs.")
@common.common_options()
def model_logs(
    remote: Optional[str], model_id: str, deployment_id: str, tail: bool = False
) -> None:
    """
    Fetches logs for the packaged model
    """

    if not remote:
        remote = remote_cli.inquire_remote_name()
    remote_provider = cast(BasetenRemote, RemoteFactory.create(remote=remote))
    if not tail:
        logs = remote_provider.api.get_model_deployment_logs(model_id, deployment_id)
        for log in cli_log_utils.parse_logs(logs):
            cli_log_utils.output_log(log)
    else:
        log_watcher = ModelDeploymentLogWatcher(
            remote_provider.api, model_id, deployment_id
        )
        for log in log_watcher.watch():
            cli_log_utils.output_log(log)


@truss_cli.command()
@click.argument("target_directory", required=False, default=os.getcwd())
@click.option(
    "--config",
    type=click.Path(exists=True),
    required=False,
    help="Path to a custom config file (default: config.yaml in truss directory)",
)
@click.option(
    "--remote",
    type=str,
    required=False,
    help="Name of the remote in .trussrc to patch changes to.",
)
@click.option(
    "--team",
    "provided_team_name",
    type=str,
    required=False,
    help="Team name for the model to watch",
)
@click.option(
    "--no-sleep",
    is_flag=True,
    default=False,
    help="Keep the development model warm by preventing scale-to-zero while watching.",
)
@click.option(
    "--hot-reload",
    "hot_reload",
    is_flag=True,
    default=False,
    help=(
        "Enable hot-reload for model code changes. "
        "Swaps model class in-process without restarting, preserving state."
    ),
)
@click.option(
    "--model-name",
    type=str,
    required=False,
    help="Temporarily override the name of the model",
)
@click.option("--tail", is_flag=True, help="Tail logs while watching.")
@common.common_options()
def watch(
    target_directory: str,
    config: Optional[str],
    remote: str,
    provided_team_name: Optional[str] = None,
    no_sleep: bool = False,
    hot_reload: bool = False,
    model_name: Optional[str] = None,
    tail: bool = False,
) -> None:
    """
    Seamless remote development with truss

    TARGET_DIRECTORY: A Truss directory. If none, use current directory.
    """
    # TODO: ensure that provider support draft
    if not remote:
        remote = remote_cli.inquire_remote_name()

    remote_provider = cast(BasetenRemote, RemoteFactory.create(remote=remote))

    tr = _get_truss_from_directory(target_directory=target_directory, config=config)
    model_name = model_name or tr.spec.config.model_name
    if not model_name:
        console.print(
            "🧐 No model name provided. Either set model_name in config.yaml or use --model-name."
        )
        sys.exit(1)

    # Resolve the model once with team disambiguation (prompts only once if needed)
    # Use config team as fallback if --team not provided
    effective_team_name = provided_team_name or RemoteFactory.get_remote_team(remote)
    resolved_model, versions = resolve_model_for_watch(
        remote_provider, model_name, provided_team_name=effective_team_name
    )
    model_id = resolved_model["id"]

    # Verify dev version exists
    dev_version = get_dev_version_from_versions(versions)
    if not dev_version:
        console.print(
            "❌ No development model found. Run `truss push --watch` then try again."
        )
        sys.exit(1)

    # Use model_id to get service (no additional resolution needed)
    dev_version_id = dev_version["id"]
    logs_url = URLConfig.model_logs_url(
        remote_provider.remote_url, model_id, dev_version_id
    )
    console.print(
        f"🪵  View logs for your development model at {common.format_link(logs_url)}"
    )

    model_hostname = resolved_model.get("hostname")
    if not model_hostname:
        console.print("❌ Could not determine model hostname", style="red")
        sys.exit(1)

    api_key = remote_provider._auth_service.authenticate().value

    common.wait_for_development_model_ready(
        model_hostname=model_hostname,
        model_id=model_id,
        dev_version_id=dev_version_id,
        remote_provider=remote_provider,
        console=console,
        api_key=api_key,
    )

    if no_sleep:
        common.start_keepalive(model_hostname, api_key)

    if tail:
        _start_tail(remote_provider, model_id, dev_version_id, in_background=True)

    # Re-resolve the model to get the latest version and truss hash and latest push before watching
    resolved_model, versions = resolve_model_for_watch(
        remote_provider, model_name, provided_team_name=effective_team_name
    )
    _start_watch_mode(
        target_directory=target_directory,
        model_name=model_name,
        remote_provider=remote_provider,
        resolved_model=resolved_model,
        resolved_versions=versions,
        console=console,
        error_console=error_console,
        hot_reload=hot_reload,
    )


### Image commands. ####################################################################


@click.group()
def image():
    """Subcommands for truss image"""


truss_cli.add_command(image)


@image.command()  # type: ignore
@click.argument("build_dir")
@click.argument("target_directory", required=False)
@common.common_options()
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
@click.option(
    "--use_host_network",
    is_flag=True,
    default=False,
    help="Use host network for docker build",
)
@common.common_options()
def build(target_directory: str, build_dir: Path, tag, use_host_network) -> None:
    """
    Builds the docker image for a Truss.

    TARGET_DIRECTORY: A Truss directory. If none, use current directory.

    BUILD_DIR: Image context. If none, a temp directory is created.
    """
    tr = _get_truss_from_directory(target_directory=target_directory)
    if build_dir:
        build_dir = Path(build_dir)
    if use_host_network:
        tr.build_serving_docker_image(build_dir=build_dir, tag=tag, network="host")
        return
    tr.build_serving_docker_image(build_dir=build_dir, tag=tag)


@image.command()  # type: ignore
@click.argument("target_directory", required=False)
@click.argument("build_dir", required=False)
@click.option("--tag", help="Docker build image tag")
@click.option("--port", type=int, default=8080, help="Local port used to run image")
@click.option(
    "--attach", is_flag=True, default=False, help="Flag for attaching the process"
)
@click.option(
    "--use_host_network",
    is_flag=True,
    default=False,
    help="Use host network for docker build",
)
@common.common_options()
def run(
    target_directory: str, build_dir: Path, tag, port, attach, use_host_network
) -> None:
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
    if use_host_network:
        tr.docker_run(
            build_dir=build_dir,
            tag=tag,
            local_port=port,
            detach=not attach,
            network="host",
        )
        return
    tr.docker_run(build_dir=build_dir, tag=tag, local_port=port, detach=not attach)


# Container commands. ##################################################################


@click.group()
def container():
    """Subcommands for truss container"""


truss_cli.add_command(container)


@container.command()  # type: ignore
@click.argument("target_directory", required=False)
@common.common_options()
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
    docker.kill_all()


# These imports are needed to register the subcommands
from truss.cli import chains_commands, migrate_commands, train_commands  # noqa: F401
