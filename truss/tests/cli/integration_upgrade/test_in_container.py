#!/usr/bin/env python
import json
import os
import pathlib
import shutil
import subprocess
import sys

RESULTS = []
TEST_VERSION = os.environ.get("TEST_VERSION", "0.12.6rc505")


def log(message, status="INFO"):
    prefix = {"INFO": "  ", "PASS": "✅", "FAIL": "❌", "WARN": "⚠️ "}[status]
    print(f"{prefix} {message}")


def run_test(name, test_fn):
    try:
        result = test_fn()
        if result:
            log(f"{name}: {result}", "PASS")
            RESULTS.append({"test": name, "status": "pass", "detail": result})
        else:
            log(f"{name}: returned falsy", "FAIL")
            RESULTS.append({"test": name, "status": "fail", "detail": "returned falsy"})
    except Exception as e:
        log(f"{name}: {e}", "FAIL")
        RESULTS.append({"test": name, "status": "fail", "detail": str(e)})


def test_truss_version():
    result = subprocess.run(
        ["truss", "--version"], capture_output=True, text=True, check=True
    )
    version = result.stdout.strip()
    assert TEST_VERSION in version, f"Expected {TEST_VERSION}, got {version}"
    return version


def test_detect_installation_method():
    from truss.cli.utils.self_upgrade import detect_installation_method

    result = detect_installation_method()
    if result is None:
        return "Detection returned None (expected for some envs)"
    method, cmd, version_fmt = result
    return f"method={method}, cmd={cmd}"


def test_upgrade_command_format():
    from truss.cli.utils.self_upgrade import detect_installation_method

    result = detect_installation_method()
    if result is None:
        raise AssertionError("Could not detect installation method")
    method, cmd, version_fmt = result
    test_cmd = cmd + version_fmt.format(version="1.2.3")
    assert "1.2.3" in test_cmd, f"Version not in command: {test_cmd}"
    return f"Formatted: {test_cmd}"


def test_notify_with_settings_enabled():
    settings_dir = pathlib.Path.home() / ".config" / "truss"
    settings_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy("/tests/settings_enabled.toml", settings_dir / "settings.toml")

    import importlib

    import truss.util.user_config as uc

    importlib.reload(uc)

    assert uc.settings.check_for_updates is True, "check_for_updates should be True"
    return "check_for_updates=True loaded correctly"


def test_notify_with_settings_disabled():
    settings_dir = pathlib.Path.home() / ".config" / "truss"
    settings_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy("/tests/settings_disabled.toml", settings_dir / "settings.toml")

    import importlib

    import truss.util.user_config as uc

    importlib.reload(uc)

    assert uc.settings.check_for_updates is False, "check_for_updates should be False"
    return "check_for_updates=False loaded correctly"


def test_upgrade_dry_run():
    from truss.cli.utils.self_upgrade import detect_installation_method

    result = detect_installation_method()
    if result is None:
        return "Skipped: detection returned None"
    method, cmd, _ = result
    print(f"  Would run: {cmd}")
    return f"Dry run prepared for {method}"


def test_actual_upgrade():
    skip_actual = os.environ.get("SKIP_ACTUAL_UPGRADE", "0") == "1"
    if skip_actual:
        return "Skipped: SKIP_ACTUAL_UPGRADE=1"

    from truss.cli.utils.self_upgrade import detect_installation_method

    result = detect_installation_method()
    if result is None:
        return "Skipped: detection returned None"

    before_version = subprocess.run(
        ["truss", "--version"], capture_output=True, text=True, check=True
    ).stdout.strip()
    print(f"  Version BEFORE: {before_version}")

    method, cmd, _ = result
    print(f"  Running: {cmd}")
    print("  " + "-" * 50)
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if proc.stdout:
        for line in proc.stdout.strip().split("\n"):
            print(f"  | {line}")
    if proc.stderr:
        for line in proc.stderr.strip().split("\n"):
            print(f"  | {line}")
    print("  " + "-" * 50)

    if proc.returncode != 0:
        raise AssertionError(f"Upgrade failed with exit code {proc.returncode}")

    after_version = subprocess.run(
        ["truss", "--version"], capture_output=True, text=True, check=True
    ).stdout.strip()
    print(f"  Version AFTER: {after_version}")

    if before_version == after_version:
        print("  ⚠️  WARNING: Version unchanged! Upgrade may not have worked.")

    return f"{before_version} → {after_version}"


def main():
    print("=" * 60)
    print("Self-Upgrade Integration Tests")
    print(f"Python: {sys.executable}")
    print(f"Prefix: {sys.prefix}")
    print(f"CONDA_PREFIX: {os.environ.get('CONDA_PREFIX', 'not set')}")
    print("=" * 60)

    run_test("truss_version", test_truss_version)
    run_test("detect_installation_method", test_detect_installation_method)
    run_test("upgrade_command_format", test_upgrade_command_format)
    run_test("settings_enabled", test_notify_with_settings_enabled)
    run_test("settings_disabled", test_notify_with_settings_disabled)
    run_test("upgrade_dry_run", test_upgrade_dry_run)
    run_test("actual_upgrade", test_actual_upgrade)

    print("=" * 60)
    passed = sum(1 for r in RESULTS if r["status"] == "pass")
    failed = sum(1 for r in RESULTS if r["status"] == "fail")
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    print("\nJSON Results:")
    print(json.dumps(RESULTS, indent=2))

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
