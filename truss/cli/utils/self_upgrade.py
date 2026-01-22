import os
import pathlib
import subprocess
import sys
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from typing import Optional

from truss.cli.utils.output import console
from truss.util import user_config


@dataclass
class InstallationInfo:
    method: str
    upgrade_command: str
    version_suffix: str


def _get_installer_info() -> Optional[tuple[str, Path]]:
    try:
        dist = distribution("truss")
        installer = (dist.read_text("INSTALLER") or "pip").strip().lower()
        dist_path = Path(str(dist.locate_file(""))).resolve()
        return (installer, dist_path)
    except PackageNotFoundError:
        return None


def detect_installation_method() -> Optional[InstallationInfo]:
    prefix = sys.prefix

    # Check for conda environment - use pip since truss isn't on conda channels
    if os.environ.get("CONDA_PREFIX") == prefix:
        return InstallationInfo(
            method="conda",
            upgrade_command=f"{sys.executable} -m pip install --upgrade truss",
            version_suffix="=={version}",
        )

    installer_info = _get_installer_info()
    if installer_info:
        installer, dist_path = installer_info

        # UV: check installer metadata
        if installer == "uv":
            if "tool" in str(dist_path).lower():
                # uv tool upgrade doesn't work when installed with exact version pin
                # Use uv tool install --force to reinstall/upgrade properly
                return InstallationInfo(
                    method="uv",
                    upgrade_command="uv tool install --force truss",
                    version_suffix="@{version}",
                )
            return InstallationInfo(
                method="uv",
                upgrade_command="uv pip install --upgrade truss",
                version_suffix="=={version}",
            )

        # PIPX: check installer metadata or path
        if installer == "pipx" or "pipx" in str(dist_path).lower():
            return InstallationInfo(
                method="pipx",
                upgrade_command="pipx upgrade truss",
                version_suffix="=={version}",
            )

    # Check for venv environment...
    pyvenv_cfg = pathlib.Path(prefix) / "pyvenv.cfg"
    if pyvenv_cfg.exists():
        return InstallationInfo(
            method="pip",
            upgrade_command=f"{sys.executable} -m pip install --upgrade truss",
            version_suffix="=={version}",
        )

    # Can be: system python, homebrew python, pyenv without venv, etc.
    # We can't confidently upgrade these, so return None.
    return None


def run_upgrade(target_version: Optional[str] = None, interactive: bool = True) -> None:
    info = detect_installation_method()

    if info is None:
        console.print(
            "Could not detect how truss was installed. Please upgrade manually.",
            style="yellow",
        )
        sys.exit(1)

    # We optionally allow specifying a version to upgrade to.
    if target_version:
        cmd = info.upgrade_command + info.version_suffix.format(version=target_version)
    else:
        cmd = info.upgrade_command

    console.print(f"Detected installation method: {info.method}")
    console.print(f"Will run: {cmd}")

    if interactive:
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
    if not user_config.settings.check_for_updates:
        return

    update_info = user_config.state.should_notify_upgrade(current_version)
    if not update_info:
        return

    latest = update_info.latest_version
    console.print(
        f"▪▪▪▪ There's a new version of truss available, {latest} "
        f"(you are currently on {current_version})!"
    )
    console.print(
        "▪▪▪▪ To upgrade to the latest version, run: [bold cyan]truss upgrade[/bold cyan]"
    )
    settings_path = user_config._SettingsWrapper.path()
    console.print(
        f"▪▪▪▪ To disable this check, set `check_for_updates` to false in {settings_path}"
    )
