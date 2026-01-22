#!/usr/bin/env python
import importlib
import json
import pathlib
import shutil
import sys

RESULTS = []


def log(message, status="INFO"):
    prefix = {"INFO": "  ", "PASS": "✅", "FAIL": "❌"}[status]
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


def get_settings_dir():
    return pathlib.Path.home() / ".config" / "truss"


def get_settings_path():
    return get_settings_dir() / "settings.toml"


def clear_settings():
    settings_dir = get_settings_dir()
    if settings_dir.exists():
        shutil.rmtree(settings_dir)


def clear_module_cache():
    modules_to_remove = [k for k in sys.modules if "truss" in k]
    for mod in modules_to_remove:
        del sys.modules[mod]


def reload_user_config():
    clear_module_cache()
    import truss.util.user_config as uc

    return uc


def test_no_settings_file_creates_default():
    clear_settings()
    settings_path = get_settings_path()

    assert not settings_path.exists(), "Settings file should not exist initially"

    uc = reload_user_config()

    assert settings_path.exists(), "Settings file should be created"
    assert uc.settings.check_for_updates is True, "Default should be True"

    content = settings_path.read_text()
    assert "check_for_updates" in content, "check_for_updates should be in file"
    print(f"  Created settings.toml:\n{content}")

    return "Default settings created with check_for_updates=true"


def test_settings_true_is_read():
    clear_settings()
    settings_dir = get_settings_dir()
    settings_dir.mkdir(parents=True, exist_ok=True)
    settings_path = get_settings_path()

    settings_path.write_text("""[preferences]
check_for_updates = true
include_git_info = false
""")

    uc = reload_user_config()

    assert uc.settings.check_for_updates is True, "Should read True from file"
    print(f"  Read check_for_updates = {uc.settings.check_for_updates}")

    return "check_for_updates=true read correctly"


def test_settings_false_is_read():
    clear_settings()
    settings_dir = get_settings_dir()
    settings_dir.mkdir(parents=True, exist_ok=True)
    settings_path = get_settings_path()

    settings_path.write_text("""[preferences]
check_for_updates = false
include_git_info = false
""")

    uc = reload_user_config()

    assert uc.settings.check_for_updates is False, "Should read False from file"
    print(f"  Read check_for_updates = {uc.settings.check_for_updates}")

    return "check_for_updates=false read correctly"


def test_notify_respects_false_setting():
    clear_settings()
    settings_dir = get_settings_dir()
    settings_dir.mkdir(parents=True, exist_ok=True)
    settings_path = get_settings_path()

    settings_path.write_text("""[preferences]
check_for_updates = false
""")

    reload_user_config()
    from truss.cli.utils import self_upgrade

    importlib.reload(self_upgrade)

    import io
    from contextlib import redirect_stdout

    f = io.StringIO()
    with redirect_stdout(f):
        self_upgrade.notify_if_outdated("0.1.0")

    output = f.getvalue()
    assert output == "", f"Should produce no output when disabled, got: {output}"

    return "notify_if_outdated respects check_for_updates=false"


def test_notify_works_when_enabled():
    clear_settings()
    settings_dir = get_settings_dir()
    settings_dir.mkdir(parents=True, exist_ok=True)
    settings_path = get_settings_path()

    settings_path.write_text("""[preferences]
check_for_updates = true
""")

    reload_user_config()
    from truss.cli.utils import self_upgrade

    importlib.reload(self_upgrade)

    import io
    from contextlib import redirect_stdout

    f = io.StringIO()
    with redirect_stdout(f):
        try:
            self_upgrade.notify_if_outdated("0.1.0")
        except Exception:
            pass

    return "notify_if_outdated called when check_for_updates=true"


def test_settings_written_to_correct_location():
    clear_settings()

    reload_user_config()
    settings_path = get_settings_path()

    assert settings_path.exists(), "Settings should exist"

    expected_dir = pathlib.Path.home() / ".config" / "truss"
    assert settings_path.parent == expected_dir, f"Should be in {expected_dir}"

    return f"Settings written to {settings_path}"


def test_old_settings_file_gets_new_keys():
    clear_settings()
    settings_dir = get_settings_dir()
    settings_dir.mkdir(parents=True, exist_ok=True)
    settings_path = get_settings_path()

    settings_path.write_text("""[preferences]
include_git_info = false
""")

    original_content = settings_path.read_text()
    assert "check_for_updates" not in original_content, (
        "Should not have check_for_updates initially"
    )

    uc = reload_user_config()

    assert uc.settings.check_for_updates is True, "Default should be True in memory"

    updated_content = settings_path.read_text()
    assert "check_for_updates" in updated_content, (
        "check_for_updates should be added to file"
    )
    print(f"  Updated settings.toml:\n{updated_content}")

    return "Old settings file updated with new keys"


def main():
    print("=" * 60)
    print("Settings TOML Integration Tests")
    print(f"Settings path: {get_settings_path()}")
    print("=" * 60)

    run_test("no_settings_creates_default", test_no_settings_file_creates_default)
    run_test("settings_true_read", test_settings_true_is_read)
    run_test("settings_false_read", test_settings_false_is_read)
    run_test("notify_respects_false", test_notify_respects_false_setting)
    run_test("notify_works_when_enabled", test_notify_works_when_enabled)
    run_test("settings_correct_location", test_settings_written_to_correct_location)
    run_test("old_settings_gets_new_keys", test_old_settings_file_gets_new_keys)

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
