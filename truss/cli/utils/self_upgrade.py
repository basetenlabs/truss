import logging
import os
import pathlib
import shutil
import subprocess
import sys

import rich.console
import rich.prompt
from InquirerPy import inquirer

from truss.cli.utils import user_config


def _run_upgrade(command: str, console: rich.console.Console) -> bool:
    console.print(f"[bold]Running:[/bold] '{command}'")
    returncode = subprocess.run(command, shell=True).returncode
    if returncode == 0:
        console.print("[green]✅ Upgrade complete. Please re-run your command.[/green]")
        return True
    else:
        console.print(
            f"[bold red]😤 Command failed with exit code {returncode}. "
            "Try upgrading manually.[/bold red]"
        )
        return False


def _is_poetry_venv() -> bool:
    # Typical pattern: poetry puts a `pyvenv.cfg` file inside the venv root
    venv_cfg_path = pathlib.Path(sys.prefix) / "pyvenv.cfg"
    if not venv_cfg_path.exists():
        return False

    text = venv_cfg_path.read_text().lower()
    return "poetry" in text or "virtualenv" in text


def _make_upgrade_command_candidates(latest_version: str) -> list[tuple[str, str]]:
    candidates = []
    if "CONDA_PREFIX" in os.environ:
        candidates.append(("conda", f"conda install truss={latest_version}"))
    if shutil.which("pipx"):
        candidates.append(("pipx", "pipx upgrade truss"))
    if _is_poetry_venv():
        candidates.append(("poetry", f"poetry add truss>={latest_version}"))
    # Pip fallback.
    candidates.append(
        ("pip", f"python -m pip install --upgrade truss=={latest_version}")
    )
    if shutil.which("pipenv"):
        candidates.append(("pipenv", "pipenv update truss"))
    if shutil.which("pdm"):
        candidates.append(("pdm", f"pdm add truss=={latest_version}"))
    if shutil.which("hatch"):
        candidates.append(("hatch", f"hatch dep add truss@{latest_version}"))
    if shutil.which("uv"):
        candidates.append(("uv", f"uv pip install --upgrade truss=={latest_version}"))
    if shutil.which("rye"):
        candidates.append(("rye", f"rye add truss@{latest_version}"))
    return candidates


def upgrade_dialogue(current_version: str, console: rich.console.Console) -> None:
    settings_wrapper = user_config.SettingsWrapper.read_or_create()
    state_wrapper = user_config.StateWrapper.read_or_create()
    update_info = state_wrapper.should_upgrade(current_version)
    latest_version = str(update_info.latest_version)
    logging.debug(f"Truss package update info: {update_info}")
    if not update_info.upgrade_recommended:
        return

    if auto_upgrade_command_template := settings_wrapper.auto_upgrade_command_template:
        console.print(
            f"[bold yellow]🪄 Automatically upgrading truss to '{latest_version}'.[/bold yellow]"
        )
        command = auto_upgrade_command_template.format(
            version=update_info.latest_version
        )
        if _run_upgrade(command, console):
            sys.exit(0)
        else:
            console.print(
                f"[bold]🖊️  You can edit or remove 'auto_upgrade_command_template' in '{settings_wrapper.path()}'[/bold]"
            )
            sys.exit(1)

    console.print(
        f"[bold yellow]⬆️  Please upgrade truss. {update_info.reason} → new version "
        f"✨'{latest_version}'✨.[/bold yellow]"
    )

    candidates = _make_upgrade_command_candidates(latest_version)
    do_nothing_cmd = "Do nothing."
    candidates.append(("🫥", do_nothing_cmd))
    options = [f"[{env}] {cmd}" for env, cmd in candidates]

    selection = inquirer.select(
        message="Pick a command for upgrading (you can edit before running):",
        choices=options,
        default=options[0],
    ).execute()

    selected_cmd = next(cmd for label, cmd in candidates if f"[{label}]" in selection)
    if selected_cmd == do_nothing_cmd:
        return

    edited_cmd = inquirer.text(
        message="🖊️  Optionally edit:", default=selected_cmd
    ).execute()

    if inquirer.confirm(
        message="▶️  Run command in this shell?", default=True
    ).execute():
        if _run_upgrade(edited_cmd, console):
            settings_path = settings_wrapper.path()
            if inquirer.confirm(
                message="💾 Do you want next time to run the same command automatically "
                f"(with newer version)? After memorizing the command, you can edit the "
                f"it in '{settings_path}'?",
                default=True,
            ).execute():
                template = edited_cmd.replace(f"{latest_version}", "{version}")
                settings_wrapper.auto_upgrade_command_template = template
            sys.exit(0)
        else:
            sys.exit(1)
