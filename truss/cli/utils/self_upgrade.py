import os
import pathlib
import subprocess
import sys
from typing import Optional

from truss.cli.utils.output import console
from truss.util import user_config


def detect_installation_method() -> Optional[tuple[str, str, str]]:
    """Returns (method_name, base_command, version_format) or None.

    version_format uses {version} as a placeholder.
    """
    prefix = sys.prefix

    # Check for conda environment - use pip since truss isn't on conda channels
    if os.environ.get("CONDA_PREFIX") == prefix:
        return (
            "conda",
            f"{sys.executable} -m pip install --upgrade truss",
            "=={version}",
        )

    # Check for venv environment...
    pyvenv_cfg = pathlib.Path(prefix) / "pyvenv.cfg"
    if pyvenv_cfg.exists():
        cfg_text = pyvenv_cfg.read_text().lower()
        # ... with uv
        if "uv" in cfg_text:
            # uv venvs don't have pip, use uv pip instead
            return ("uv", "uv pip install --upgrade truss", "=={version}")
        # ... with pip
        return (
            "pip",
            f"{sys.executable} -m pip install --upgrade truss",
            "=={version}",
        )

    # Can be: system python, homebrew python, pyenv without venv, pipx, etc.
    # We can't confidently upgrade these, so return None.
    return None


def run_upgrade(target_version: Optional[str] = None) -> None:
    result = detect_installation_method()

    if result is None:
        console.print(
            "Could not detect how truss was installed. Please upgrade manually:",
            style="yellow",
        )
        console.print("  pip install --upgrade truss")
        sys.exit(1)

    method, base_cmd, version_fmt = result

    # We optionally allow specifying a version to upgrade to.
    if target_version:
        cmd = base_cmd + version_fmt.format(version=target_version)
    else:
        cmd = base_cmd

    console.print(f"Detected installation method: {method}")
    console.print(f"Will run: {cmd}")

    response = console.input("Proceed with upgrade? [Y/n] ")
    if response.lower() not in ("", "y", "yes"):
        console.print("Upgrade cancelled.")
        return

    returncode = subprocess.run(cmd, shell=True).returncode
    if returncode == 0:
        console.print("✅ Upgrade complete.", style="green")
    else:
        console.print(f"❌ Upgrade failed (exit code {returncode})", style="red")
        sys.exit(1)


def notify_if_outdated(current_version: str) -> None:
    update_info = user_config.state.should_notify_upgrade(current_version)
    if not update_info:
        return

    latest = update_info.latest_version
    console.print(
        f"A newer truss version {latest} is available. You're on {current_version}."
    )
    user_config.state.mark_notified(latest)


def prompt_upgrade_if_outdated(current_version: str) -> None:
    update_info = user_config.state.should_notify_upgrade(current_version)
    if not update_info:
        return

    latest = update_info.latest_version

    if detect_installation_method() is None:
        console.print(
            f"A newer truss version {latest} is available. You're on {current_version}."
        )
        user_config.state.mark_notified(latest)
        return

    response = console.input(
        f"A newer truss version {latest} is available. "
        f"You're on {current_version}. Upgrade now? [Y/n] "
    )

    user_config.state.mark_notified(latest)

    if response.lower() in ("", "y", "yes"):
        # Note: we run upgrade and then exit.
        # user will have to re-run their original command
        run_upgrade()
        # Note: reached only if upgrade was successful
        # but we want to exit the old process now
        sys.exit(0)
