import time
from pathlib import Path
from typing import List, Optional, Tuple

import rich
import rich.live
import rich.logging
import rich.spinner
import rich.table
import rich.traceback
import rich_click as click
from InquirerPy import inquirer
from rich import progress

from truss.cli import remote_cli
from truss.cli.cli import truss_cli
from truss.cli.utils import common, output
from truss.cli.utils.output import console
from truss.remote.baseten.core import ACTIVE_STATUS, DEPLOYING_STATUSES
from truss.remote.baseten.utils.status import get_displayable_status
from truss.util import user_config
from truss.util.log_utils import LogInterceptor

# Chains Stuff #########################################################################


@click.group()
def chains():
    """Subcommands for truss chains"""


truss_cli.add_command(chains)


def _load_example_chainlet_code() -> str:
    try:
        from truss_chains.reference_code import reference_chainlet
    # if the example is faulty, a validation error would be raised
    except Exception as e:
        raise Exception("Failed to load starter code. Please notify support.") from e

    source = Path(reference_chainlet.__file__).read_text()
    return source


def _make_chains_curl_snippet(run_remote_url: str, environment: Optional[str]) -> str:
    if environment:
        idx = run_remote_url.find("deployment")
        if idx != -1:
            run_remote_url = (
                run_remote_url[:idx] + f"environments/{environment}/run_remote"
            )
    return (
        f"curl -X POST '{run_remote_url}' \\\n"
        '    -H "Authorization: Api-Key $BASETEN_API_KEY" \\\n'
        "    -d '<JSON_INPUT>'"
    )


def _create_chains_table(service) -> Tuple[rich.table.Table, List[str]]:
    """Creates a status table similar to:

                                          ⛓️   ItestChain - Chain  ⛓️

                         🌐 Status page: https://app.baseten.co/chains/p7qrm93v/overview
    ╭──────────────────────┬──────────────────────────────┬─────────────────────────────────────────────╮
    │ Status               │ Chainlet                     │ Logs URL                                   │
    ├──────────────────────┼──────────────────────────────┼────────────────────────────────────────────┤
    │ 🛠️  BUILDING         │ ItestChain (entrypoint)      │ https://app.baseten.co/chains/.../logs/... │
    ├──────────────────────┼──────────────────────────────┼────────────────────────────────────────────┤
    │ 👾  DEPLOYING        │ GENERATE_DATA (internal)     │ https://app.baseten.co/chains/.../logs/... │
    │ 👾  DEPLOYING        │ SplitTextFailOnce (internal) │ https://app.baseten.co/chains/.../logs/... │
    │ 👾  DEPLOYING        │ TextReplicator (internal)    │ https://app.baseten.co/chains/.../logs/... │
    │ 🛠️  BUILDING         │ TextToNum (internal)         │ https://app.baseten.co/chains/.../logs/... │
    ╰──────────────────────┴──────────────────────────────┴────────────────────────────────────────────╯

    """
    title = (
        f"⛓️   {service.name} - Chain  ⛓️\n\n "
        f"🌐 Status page: {common.format_link(service.status_page_url)}"
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
    table.add_column("Logs UI")
    statuses = []
    status_iterable = service.get_info()
    # Organize status_iterable s.t. entrypoint is first.
    entrypoint = next(x for x in status_iterable if x.is_entrypoint)
    sorted_chainlets = sorted(
        (x for x in status_iterable if not x.is_entrypoint), key=lambda x: x.name
    )
    for i, chainlet in enumerate([entrypoint] + sorted_chainlets):
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

        table.add_row(
            spinner, display_name, common.format_link(chainlet.logs_url, "click here")
        )
        # Add section divider after entrypoint, entrypoint must be first.
        if chainlet.is_entrypoint:
            table.add_section()
        statuses.append(displayable_status)
    return table, statuses


@chains.command(name="push")  # type: ignore
@click.argument("source", type=Path, required=True)
@click.argument("entrypoint", type=str, required=False)
@click.option(
    "--name",
    type=str,
    required=False,
    help="Name of the chain to be deployed, if not given, the entrypoint name is used.",
)
@click.option(
    "--publish/--no-publish",
    default=False,
    help="Create chainlets as published deployments.",
)
@click.option(
    "--promote/--no-promote",
    default=False,
    help="Replace production chainlets with newly deployed chainlets.",
)
@click.option(
    "--environment",
    type=str,
    required=False,
    help=(
        "Deploy the chain as a published deployment to the specified environment."
        "If specified, --publish is implied and the supplied value of --promote will be ignored."
    ),
)
@click.option(
    "--wait/--no-wait",
    default=True,
    help="Wait until all chainlets are ready (or deployment failed).",
)
@click.option(
    "--watch/--no-watch",
    default=False,
    help=(
        "Watches the chains source code and applies live patches. Using this option "
        "will wait for the chain to be deployed (i.e. `--wait` flag is applied), "
        "before starting to watch for changes. This option required the deployment "
        "to be a development deployment (i.e. `--no-promote` and `--no-publish`."
    ),
)
@click.option(
    "--dryrun",
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
@click.option(
    "--experimental-watch-chainlet-names",
    type=str,
    required=False,
    help=(
        "Runs `watch`, but only applies patches to specified chainlets. The option is "
        "a comma-separated list of chainlet (display) names. This option can give "
        "faster dev loops, but also lead to inconsistent deployments. Use with caution "
        "and refer to docs."
    ),
)
@click.option(
    "--include-git-info",
    is_flag=True,
    required=False,
    default=False,
    help=common.INCLUDE_GIT_INFO_DOC,
)
@click.pass_context
@common.common_options()
def push_chain(
    ctx: click.Context,
    source: Path,
    entrypoint: Optional[str],
    name: Optional[str],
    publish: bool,
    promote: bool,
    wait: bool,
    watch: bool,
    dryrun: bool,
    remote: Optional[str],
    environment: Optional[str],
    experimental_watch_chainlet_names: Optional[str],
    include_git_info: bool = False,
) -> None:
    """
    Deploys a chain remotely.

    SOURCE: Path to a python file that contains the entrypoint chainlet.

    ENTRYPOINT: Class name of the entrypoint chainlet in source file. May be omitted
    if a chainlet definition in SOURCE is tagged with `@chains.mark_entrypoint`.
    """
    # These imports are delayed, to handle pydantic v1 envs gracefully.
    from truss_chains import framework
    from truss_chains import private_types as chains_def
    from truss_chains.deployment import deployment_client

    if experimental_watch_chainlet_names:
        watch = True

    if watch:
        if publish or promote:
            raise ValueError(
                "When using `--watch`, the deployment cannot be published or promoted."
            )
        if not wait:
            console.print(
                "'--watch' is used. Will wait for deployment before watching files."
            )
            wait = True

    if promote and environment:
        promote_warning = (
            "'promote' flag and 'environment' flag were both specified. "
            "Ignoring the value of 'promote'."
        )
        console.print(promote_warning, style="yellow")

    if not remote:
        if dryrun:
            remote = ""
        else:
            remote = remote_cli.inquire_remote_name()

    if not include_git_info:
        include_git_info = user_config.settings.include_git_info

    with framework.ChainletImporter.import_target(source, entrypoint) as entrypoint_cls:
        chain_name = (
            name or entrypoint_cls.meta_data.chain_name or entrypoint_cls.display_name
        )
        options = chains_def.PushOptionsBaseten.create(
            chain_name=chain_name,
            promote=promote,
            publish=publish,
            only_generate_trusses=dryrun,
            remote=remote,
            environment=environment,
            include_git_info=include_git_info,
            working_dir=source.parent if source.is_file() else source.resolve(),
        )
        service = deployment_client.push(
            entrypoint_cls, options, progress_bar=progress.Progress
        )

    if dryrun:
        return

    assert isinstance(service, deployment_client.BasetenChainService)
    curl_snippet = _make_chains_curl_snippet(
        service.run_remote_url, options.environment
    )

    table, statuses = _create_chains_table(service)
    status_check_wait_sec = 2
    if wait:
        num_services = len(statuses)
        success = False
        num_failed = 0
        # Logging inferences with live display (even when using richHandler)
        # -> capture logs and print later.
        with (
            LogInterceptor() as log_interceptor,
            rich.live.Live(table, refresh_per_second=4) as live,
        ):
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

            intercepted_logs = log_interceptor.get_logs()

        # Prints must be outside `Live` context.
        if intercepted_logs:
            console.print("Logs intercepted during waiting:", style="blue")
            for log in intercepted_logs:
                console.print(f"\t{log}")
        if success:
            deploy_success_text = "Deployment succeeded."
            if environment:
                deploy_success_text = (
                    "Your chain has been deployed into "
                    f"the {options.environment} environment."
                )
            console.print(deploy_success_text, style="bold green")
            console.print(f"You can run the chain with:\n{curl_snippet}")

            if watch:  # Note that this command will print a startup message.
                if experimental_watch_chainlet_names:
                    included_chainlets = [
                        x.strip() for x in experimental_watch_chainlet_names.split(",")
                    ]
                else:
                    included_chainlets = None
                deployment_client.watch(
                    source,
                    entrypoint,
                    name,
                    remote,
                    rich.Console(),
                    output.error_console,
                    show_stack_trace=not common.is_human_log_level(ctx),
                    included_chainlets=included_chainlets,
                )
        else:
            console.print(f"Deployment failed ({num_failed} failures).", style="red")
    else:
        console.print(table)
        console.print(
            "Once all chainlets are deployed, "
            f"you can run the chain with:\n\n{curl_snippet}"
        )


@chains.command(name="watch")  # type: ignore
@click.argument("source", type=Path, required=True)
@click.argument("entrypoint", type=str, required=False)
@click.option(
    "--name",
    type=str,
    required=False,
    help="Name of the chain to be deployed, if not given, the entrypoint name is used.",
)
@click.option(
    "--remote",
    type=str,
    required=False,
    help="Name of the remote in .trussrc to push to.",
)
@click.option(
    "--experimental-chainlet-names",
    type=str,
    required=False,
    help=(
        "Runs 'watch', but only applies patches to specified chainlets. The option is "
        "a comma-separated list of chainlet (display) names. This option can give "
        "faster dev loops, but also lead to inconsistent deployments. Use with caution "
        "and refer to docs."
    ),
)
@click.pass_context
@common.common_options()
def watch_chains(
    ctx: click.Context,
    source: Path,
    entrypoint: Optional[str],
    name: Optional[str],
    remote: Optional[str],
    experimental_chainlet_names: Optional[str],
) -> None:
    """
    Watches the chains source code and applies live patches to a development deployment.

    The development deployment must have been deployed before running this command.

    SOURCE: Path to a python file that contains the entrypoint chainlet.

    ENTRYPOINT: Class name of the entrypoint chainlet in source file. May be omitted
    if a chainlet definition in SOURCE is tagged with `@chains.mark_entrypoint`.
    """
    # These imports are delayed, to handle pydantic v1 envs gracefully.
    from truss_chains.deployment import deployment_client

    if not remote:
        remote = remote_cli.inquire_remote_name()

    if experimental_chainlet_names:
        included_chainlets = [x.strip() for x in experimental_chainlet_names.split(",")]
    else:
        included_chainlets = None

    deployment_client.watch(
        source,
        entrypoint,
        name,
        remote,
        rich.Console(),
        output.error_console,
        show_stack_trace=not common.is_human_log_level(ctx),
        included_chainlets=included_chainlets,
    )


@chains.command(name="init")  # type: ignore
@click.argument("directory", type=Path, required=False)
@common.common_options()
def init_chain(directory: Optional[Path]) -> None:
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
    console.print(f"Creating and populating {filepath}...\n")
    source_code = _load_example_chainlet_code()
    filepath.write_text(source_code)
    console.print(
        "Next steps:\n",
        f"💻 Run [bold green]'python {filepath}'[/bold green] for local debug "
        "execution.\n"
        f"🚢 Run [bold green]'truss chains push {filepath}'[/bold green] "
        "to deploy the chain to Baseten.\n",
    )
