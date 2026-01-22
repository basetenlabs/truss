#!/usr/bin/env python
import importlib
import os
import pathlib
import shutil
import subprocess
import sys

from test_harness import run_tests_and_exit

import truss.util.user_config as uc
from truss.cli.utils.self_upgrade import detect_installation_method

TEST_VERSION = os.environ.get("TEST_VERSION", "0.12.6rc505")
SETTINGS_DIR = pathlib.Path.home() / ".config" / "truss"


def get_truss_version():
    return subprocess.run(
        ["truss", "--version"], capture_output=True, text=True, check=True
    ).stdout.strip()


def test_truss_version():
    version = get_truss_version()
    assert TEST_VERSION in version, f"Expected {TEST_VERSION}, got {version}"
    return version


def test_detect_installation_method():
    result = detect_installation_method()
    if result is None:
        return "Detection returned None (expected for some envs)"
    return f"method={result.method}, cmd={result.upgrade_command}"


def test_upgrade_command_format():
    result = detect_installation_method()
    if result is None:
        raise AssertionError("Could not detect installation method")
    test_cmd = result.upgrade_command + result.version_suffix.format(version="1.2.3")
    assert "1.2.3" in test_cmd, f"Version not in command: {test_cmd}"
    return f"Formatted: {test_cmd}"


def test_settings_check_for_updates(enabled: bool):
    SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
    src = f"/tests/settings_{'enabled' if enabled else 'disabled'}.toml"
    shutil.copy(src, SETTINGS_DIR / "settings.toml")
    importlib.reload(uc)
    assert uc.settings.check_for_updates is enabled
    return f"check_for_updates={enabled} loaded correctly"


def test_upgrade_dry_run():
    result = detect_installation_method()
    if result is None:
        return "Skipped: detection returned None"
    print(f"  Would run: {result.upgrade_command}")
    return f"Dry run prepared for {result.method}"


def test_actual_upgrade():
    if os.environ.get("SKIP_ACTUAL_UPGRADE", "0") == "1":
        return "Skipped: SKIP_ACTUAL_UPGRADE=1"

    result = detect_installation_method()
    if result is None:
        return "Skipped: detection returned None"

    before = get_truss_version()
    print(f"  Version BEFORE: {before}")
    print(f"  Running: {result.upgrade_command}")
    print("  " + "-" * 50)

    proc = subprocess.run(
        result.upgrade_command, shell=True, capture_output=True, text=True
    )
    for stream in (proc.stdout, proc.stderr):
        if stream:
            for line in stream.strip().split("\n"):
                print(f"  | {line}")
    print("  " + "-" * 50)

    if proc.returncode != 0:
        raise AssertionError(f"Upgrade failed with exit code {proc.returncode}")

    after = get_truss_version()
    print(f"  Version AFTER: {after}")
    if before == after:
        print("  ⚠️  WARNING: Version unchanged! Upgrade may not have worked.")
    return f"{before} → {after}"


if __name__ == "__main__":
    print(f"Python: {sys.executable}, Prefix: {sys.prefix}")
    print(f"CONDA_PREFIX: {os.environ.get('CONDA_PREFIX', 'not set')}")

    run_tests_and_exit(
        "Self-Upgrade Integration Tests",
        [
            ("truss_version", test_truss_version),
            ("detect_installation_method", test_detect_installation_method),
            ("upgrade_command_format", test_upgrade_command_format),
            ("settings_enabled", lambda: test_settings_check_for_updates(True)),
            ("settings_disabled", lambda: test_settings_check_for_updates(False)),
            ("upgrade_dry_run", test_upgrade_dry_run),
            ("actual_upgrade", test_actual_upgrade),
        ],
    )
