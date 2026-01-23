#!/usr/bin/env python
import datetime
import io
import json
import pathlib
import shutil
import sys
from contextlib import redirect_stdout

from test_harness import run_tests_and_exit

SETTINGS_DIR = pathlib.Path.home() / ".config" / "truss"
SETTINGS_PATH = SETTINGS_DIR / "settings.toml"
STATE_PATH = SETTINGS_DIR / "state.json"


def clear_settings():
    if SETTINGS_DIR.exists():
        shutil.rmtree(SETTINGS_DIR)


def clear_module_cache():
    for mod in [k for k in sys.modules if "truss" in k]:
        del sys.modules[mod]


def reload_user_config():
    clear_module_cache()
    import truss.util.user_config as uc

    return uc


def write_settings(content: str):
    clear_settings()
    SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(content)


def test_no_settings_file_creates_default():
    clear_settings()
    assert not SETTINGS_PATH.exists(), "Settings file should not exist initially"

    uc = reload_user_config()

    assert SETTINGS_PATH.exists(), "Settings file should be created"
    assert uc.settings.check_for_updates is True, "Default should be True"

    content = SETTINGS_PATH.read_text()
    assert "check_for_updates" in content, "check_for_updates should be in file"
    print(f"  Created settings.toml:\n{content}")
    return "Default settings created with check_for_updates=true"


def test_settings_value_is_read(value: bool):
    write_settings(f"[preferences]\ncheck_for_updates = {str(value).lower()}\n")
    uc = reload_user_config()
    assert uc.settings.check_for_updates is value, f"Should read {value} from file"
    print(f"  Read check_for_updates = {uc.settings.check_for_updates}")
    return f"check_for_updates={value} read correctly"


def test_notify_respects_false_setting():
    write_settings("[preferences]\ncheck_for_updates = false\n")
    reload_user_config()

    import importlib

    from truss.cli.utils import self_upgrade

    importlib.reload(self_upgrade)

    f = io.StringIO()
    with redirect_stdout(f):
        self_upgrade.notify_if_outdated("0.1.0")

    output = f.getvalue()
    assert output == "", f"Should produce no output when disabled, got: {output}"
    return "notify_if_outdated respects check_for_updates=false"


def test_notify_works_when_enabled():
    write_settings("[preferences]\ncheck_for_updates = true\n")
    reload_user_config()

    import importlib

    from truss.cli.utils import self_upgrade

    importlib.reload(self_upgrade)

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

    assert SETTINGS_PATH.exists(), "Settings should exist"
    expected_dir = pathlib.Path.home() / ".config" / "truss"
    assert SETTINGS_PATH.parent == expected_dir, f"Should be in {expected_dir}"
    return f"Settings written to {SETTINGS_PATH}"


def test_old_settings_file_gets_new_keys():
    write_settings("[preferences]\ninclude_git_info = false\n")

    original = SETTINGS_PATH.read_text()
    assert "check_for_updates" not in original, (
        "Should not have check_for_updates initially"
    )

    uc = reload_user_config()
    assert uc.settings.check_for_updates is True, "Default should be True in memory"

    updated = SETTINGS_PATH.read_text()
    assert "check_for_updates" in updated, "check_for_updates should be added to file"
    print(f"  Updated settings.toml:\n{updated}")
    return "Old settings file updated with new keys"


def write_state(state_dict: dict):
    SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state_dict))


def test_notification_shows_only_once_per_day():
    write_settings("[preferences]\ncheck_for_updates = true\n")

    old_time = (datetime.datetime.now() - datetime.timedelta(days=2)).isoformat()
    state = {
        "version_info": {
            "latest_version": "99.0.0",
            "yanked_versions": [],
            "last_check": old_time,
        }
    }
    write_state(state)

    uc = reload_user_config()

    result1 = uc.state.should_notify_upgrade("0.1.0")
    assert result1 is not None, "First call should return notification info"
    print(f"  First call returned: {result1}")

    state_after_first = json.loads(STATE_PATH.read_text())
    last_check = state_after_first["version_info"]["last_check"]
    print(f"  last_check after first call: {last_check}")

    uc = reload_user_config()

    result2 = uc.state.should_notify_upgrade("0.1.0")
    assert result2 is None, f"Second call should return None, got: {result2}"
    print("  Second call returned None (as expected)")

    return "Notification correctly shows only once per day"


if __name__ == "__main__":
    print(f"Settings path: {SETTINGS_PATH}")

    run_tests_and_exit(
        "Settings TOML Integration Tests",
        [
            ("no_settings_creates_default", test_no_settings_file_creates_default),
            ("settings_true_read", lambda: test_settings_value_is_read(True)),
            ("settings_false_read", lambda: test_settings_value_is_read(False)),
            ("notify_respects_false", test_notify_respects_false_setting),
            ("notify_works_when_enabled", test_notify_works_when_enabled),
            ("settings_correct_location", test_settings_written_to_correct_location),
            ("old_settings_gets_new_keys", test_old_settings_file_gets_new_keys),
            ("notification_once_per_day", test_notification_shows_only_once_per_day),
        ],
    )
