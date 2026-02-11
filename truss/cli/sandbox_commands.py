import shlex
import threading
import time
from typing import Optional

import b10sb
import rich_click as click
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from truss.cli.cli import truss_cli
from truss.cli.utils import common
from truss.cli.utils.output import console

# Border styles and colors for the animated box
_SANDBOX_ANIM_BORDERS = [
    "bright_blue",
    "cyan",
    "bright_cyan",
    "blue",
    "bright_magenta",
    "magenta",
]
_SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

_STATUS_STYLES = {"running": "green", "stopped": "red", "unknown": "dim"}


def _styled_status(status: str) -> str:
    """Return status string wrapped in Rich markup for its color."""
    style = _STATUS_STYLES.get(status, "dim")
    return f"[{style}]{status}[/{style}]"


# Sand frames – the tildes slosh left then right inside the box.
# Each entry is exactly 19 chars to fit between the box walls.
_SAND_FRAMES = [
    "  ~ ~ ~ ~ ~ ~ ~ ~  ",
    " ~ ~ ~ ~ ~ ~ ~ ~   ",
    "~ ~ ~ ~ ~ ~ ~ ~    ",
    " ~ ~ ~ ~ ~ ~ ~ ~   ",
    "  ~ ~ ~ ~ ~ ~ ~ ~  ",
    "   ~ ~ ~ ~ ~ ~ ~ ~ ",
    "    ~ ~ ~ ~ ~ ~ ~ ~",
    "   ~ ~ ~ ~ ~ ~ ~ ~ ",
]


def _baseten_banner(frame: int) -> str:
    """Return the sandbox ASCII art with animated sand for the given frame."""
    sand = _SAND_FRAMES[(frame // 2) % len(_SAND_FRAMES)]
    return (
        "   __           \n"
        '  |__|  .-"""""""""""""""-.    \n'
        "   ||  /  .-.-.-.-.-.-.-.  \\  \n"
        f"  _||_ |{sand}|  \n"
        " |____||___________________|  \n"
        "\n"
    )


def _sandbox_animation_frame(frame: int, instances: int = 1) -> Panel:
    """Build one frame of the sandbox creation animation."""
    spinner = _SPINNER_FRAMES[frame % len(_SPINNER_FRAMES)]
    border_style = _SANDBOX_ANIM_BORDERS[frame % len(_SANDBOX_ANIM_BORDERS)]
    noun = "sandbox" if instances == 1 else "sandboxes"
    verb = "is" if instances == 1 else "are"

    content = Text()
    content.append(_baseten_banner(frame), style=f"bold {border_style}")
    content.append(f"  {spinner} ", style=border_style)
    content.append(f"Creating {noun}...", style="bold")
    content.append("\n\n  ", style="dim")
    content.append(f"Your {noun} {verb} spinning up.", style="dim italic")

    return Panel(
        content,
        title="[bold] Sandbox [/bold]",
        title_align="left",
        border_style=border_style,
        padding=(1, 2),
        expand=False,
        width=56,
    )


def _sandbox_complete_panel(sandbox_ids: list[str]) -> Panel:
    """Build the completion panel shown after sandbox creation."""
    content = Text()
    content.append(_baseten_banner(0), style="bold green")
    content.append("  ✓ ", style="bold green")

    if len(sandbox_ids) == 1:
        content.append("Sandbox created\n\n", style="bold")
        content.append("  ID  ", style="dim")
        content.append(sandbox_ids[0], style="bold")
    else:
        content.append(f"{len(sandbox_ids)} sandboxes created\n\n", style="bold")
        for sid in sorted(sandbox_ids):
            content.append("  • ", style="green")
            content.append(f"{sid}\n", style="bold")

    return Panel(
        content,
        title="[bold] Sandbox [/bold]",
        title_align="left",
        border_style="green",
        padding=(1, 2),
        expand=False,
        width=56,
    )


def _sandbox_action_frame(frame: int, action_label: str, subtitle: str) -> Panel:
    """Build one frame of a sandbox action animation (stop, resume, etc.)."""
    spinner = _SPINNER_FRAMES[frame % len(_SPINNER_FRAMES)]
    border_style = _SANDBOX_ANIM_BORDERS[frame % len(_SANDBOX_ANIM_BORDERS)]

    content = Text()
    content.append(f"  {spinner} ", style=border_style)
    content.append(action_label, style="bold")
    content.append("\n\n  ", style="dim")
    content.append(subtitle, style="dim italic")

    return Panel(
        content,
        title="[bold] Sandbox [/bold]",
        title_align="left",
        border_style=border_style,
        padding=(1, 2),
        expand=False,
        width=56,
    )


def _run_with_sandbox_spinner(action_fn, action_label: str, subtitle: str):
    """Run action_fn in a background thread while showing a sandbox spinner."""
    done = threading.Event()
    exception_holder: list[BaseException] = []
    result_holder: list = []

    def run():
        try:
            result_holder.append(action_fn())
        except BaseException as e:
            exception_holder.append(e)
        finally:
            done.set()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()

    frame = 0
    with Live(
        _sandbox_action_frame(0, action_label, subtitle),
        refresh_per_second=8,
        console=console,
        transient=True,
    ) as live:
        while not done.is_set():
            time.sleep(1 / 8)
            frame += 1
            live.update(_sandbox_action_frame(frame, action_label, subtitle))

    thread.join(timeout=0)
    if exception_holder:
        raise exception_holder[0]

    return result_holder[0] if result_holder else None


@click.group()
def sandbox():
    """Subcommands for truss train"""


truss_cli.add_command(sandbox)


@sandbox.command(name="create")
@click.option(
    "--instances",
    type=int,
    required=False,
    help="Number of instances to create.",
    default=1,
)
@click.option("--name", type=str, required=False, help="Name of the sandbox.")
@click.option(
    "--ports",
    "ports",
    multiple=True,
    type=int,
    help="Ports to expose on the sandbox (repeat for multiple: --ports 8000 --ports 8001).",
    default=(),
)
@common.common_options()
def create_sandbox(instances: int, name: Optional[str], ports: tuple[int, ...]):
    """Create a sandbox"""
    done = threading.Event()
    exception_holder: list[BaseException] = []

    created_sandbox_ids: list[str] = []

    def run_create():
        try:
            remote_provider = b10sb.RemoteSandboxProvider(
                api_base_url="https://dreambox.internal.basetensors.com/sandboxes"
            )
            config = b10sb.SandboxConfig(
                image="debian:bookworm-slim", expose=list(ports), name=name
            )
            for _ in range(instances):
                try:
                    sandbox = remote_provider.create(config=config)
                    created_sandbox_ids.append(sandbox.sandbox_id)
                except RuntimeError as e:
                    exception_holder.append(e)
        except BaseException as e:
            exception_holder.append(e)
        finally:
            done.set()

    thread = threading.Thread(target=run_create, daemon=True)
    thread.start()

    frame = 0
    with Live(
        _sandbox_animation_frame(0, instances),
        refresh_per_second=8,
        console=console,
        transient=True,
    ) as live:
        while not done.is_set():
            time.sleep(1 / 8)
            frame += 1
            live.update(_sandbox_animation_frame(frame, instances))

    thread.join(timeout=0)
    if exception_holder:
        raise exception_holder[0]

    ids = sorted(set(created_sandbox_ids))
    console.print(_sandbox_complete_panel(ids))
    if len(ids) == 1:
        console.print()
        console.print("  Quick start:", style="dim")
        console.print(
            f'  truss sandbox exec {ids[0]} -- echo "Hello world"', style="cyan"
        )


@sandbox.command(name="list")
@common.common_options()
def list_sandboxes():
    """List all sandboxes"""
    remote_provider = b10sb.RemoteSandboxProvider(
        api_base_url="https://dreambox.internal.basetensors.com/sandboxes"
    )
    sandboxes = remote_provider.list()
    table = Table(title="Sandboxes")
    table.add_column("ID")
    table.add_column("Name")
    table.add_column("Status")
    for sandbox in sandboxes:
        table.add_row(sandbox.sandbox_id, sandbox.name, _styled_status(sandbox.status))
    console.print(table)


def _get_sandbox_or_exit(sandbox_id: str):
    """Return RemoteSandbox for sandbox_id or print error and exit."""
    remote_provider = b10sb.RemoteSandboxProvider(
        api_base_url="https://dreambox.internal.basetensors.com/sandboxes"
    )
    sandbox = remote_provider.get(sandbox_id)
    if not sandbox:
        console.print(f"Sandbox with ID {sandbox_id} not found", style="bold red")
        raise SystemExit(1)
    return sandbox


@sandbox.command(name="exec")
@click.argument("sandbox_id", type=str, required=True)
@click.argument("command", type=str, nargs=-1, required=True)
@click.option("--timeout", type=int, default=None, help="Command timeout in seconds.")
@common.common_options()
def exec_sandbox(sandbox_id: str, command: tuple[str, ...], timeout: int | None):
    """Run a command in a sandbox and print stdout, stderr, and exit code.

    Use '--' before the command if it contains options (e.g. -- python3 -c 'print(1)').
    """
    sandbox = _get_sandbox_or_exit(sandbox_id)
    cmd_str = " ".join(shlex.quote(arg) for arg in command)
    result = sandbox.execute(cmd_str, timeout=timeout)
    if result.stdout:
        console.print(result.stdout)
    if result.stderr:
        console.print(result.stderr, style="red")
    raise SystemExit(result.exit_code)


@sandbox.command(name="exec-stream")
@click.argument("sandbox_id", type=str, required=True)
@click.argument("command", type=str, nargs=-1, required=True)
@common.common_options()
def exec_stream_sandbox(sandbox_id: str, command: tuple[str, ...]):
    """Run a command in a sandbox and stream output (SSE).

    Use '--' before the command if it contains options (e.g. -- python3 -c 'print(1)').
    """
    sandbox = _get_sandbox_or_exit(sandbox_id)
    cmd_str = " ".join(shlex.quote(arg) for arg in command)
    for chunk in sandbox.execute_stream(cmd_str):
        if chunk.get("type") == "log":
            data = chunk.get("data", "")
            console.print(data, end="" if data.endswith("\n") else "\n")
        elif chunk.get("type") == "status":
            exit_code = chunk.get("exit_code", -1)
            if exit_code != 0:
                raise SystemExit(exit_code)


@sandbox.command(name="get")
@click.argument("sandbox_id", type=str, required=True)
@common.common_options()
def get_sandbox(sandbox_id: str):
    """Get information about a sandbox"""
    remote_provider = b10sb.RemoteSandboxProvider(
        api_base_url="https://dreambox.internal.basetensors.com/sandboxes"
    )
    sandbox = remote_provider.get(sandbox_id)
    if not sandbox:
        console.print(f"Sandbox with ID {sandbox_id} not found", style="bold red")
        return
    info = sandbox.get_status()
    table = Table(title=f"Sandbox {sandbox_id}")
    table.add_column("ID")
    table.add_column("Name")
    table.add_column("Status")
    table.add_row(info.sandbox_id, info.name, _styled_status(info.status))
    console.print(table)


@sandbox.command(name="stop")
@click.argument("sandbox_id", type=str, required=True)
@common.common_options()
def stop_sandbox(sandbox_id: str):
    """Stop a sandbox"""
    remote_provider = b10sb.RemoteSandboxProvider(
        api_base_url="https://dreambox.internal.basetensors.com/sandboxes"
    )
    stopped = _run_with_sandbox_spinner(
        lambda: remote_provider.stop(sandbox_id),
        "Putting the lid on the sandbox...",
        "It'll be here when you get back.",
    )
    if stopped:
        console.print(f"Sandbox {sandbox_id} stopped", style="bold green")
    else:
        console.print(f"Sandbox {sandbox_id} not found", style="bold red")


@sandbox.command(name="delete")
@click.argument("sandbox_id", type=str, required=True)
@common.common_options()
def delete_sandbox(sandbox_id: str):
    """Permanently delete a sandbox (removes it; no longer appears in list)."""
    remote_provider = b10sb.RemoteSandboxProvider(
        api_base_url="https://dreambox.internal.basetensors.com/sandboxes"
    )
    deleted = _run_with_sandbox_spinner(
        lambda: remote_provider.delete(sandbox_id),
        "Removing the sandbox...",
        "Sandbox removed.",
    )
    if deleted:
        console.print(f"Sandbox {sandbox_id} deleted", style="bold green")
    else:
        console.print(f"Sandbox {sandbox_id} not found", style="bold red")


@sandbox.command(name="resume")
@click.argument("sandbox_id", type=str, required=True)
@common.common_options()
def resume_sandbox(sandbox_id: str):
    """Resume a sandbox"""
    remote_provider = b10sb.RemoteSandboxProvider(
        api_base_url="https://dreambox.internal.basetensors.com/sandboxes"
    )
    sandbox = remote_provider.get(sandbox_id)
    if not sandbox:
        console.print(f"Sandbox with ID {sandbox_id} not found", style="bold red")
        return
    resumed = _run_with_sandbox_spinner(
        sandbox.resume, "Dusting off the sandbox...", "Almost ready to play."
    )
    if resumed:
        console.print(f"Sandbox {sandbox_id} resumed", style="bold green")
    else:
        console.print(
            f"Could not resume {sandbox_id}; please try again or contact support.",
            style="bold red",
        )
