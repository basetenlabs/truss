import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parents[4]
SCRIPT = REPO_ROOT / "scripts" / "update_cannery_artifact_pins.py"
BUNDLED_PINS = REPO_ROOT / "truss" / "cli" / "cannery" / "bundled_artifacts.json"


def _pin(**updates):
    pin = {
        "cannery_version": "1.2.3",
        "protocol_version": 1,
        "operating_system": "linux",
        "architecture": "x86_64",
        "url": "https://baseten-public.s3.us-west-2.amazonaws.com/bin/cannery/1.2.3/cannery-1.2.3-linux-x86_64",
        "size_bytes": 123,
        "sha256": "a" * 64,
    }
    pin.update(updates)
    return pin


def test_release_pin_imports_exact_baseten_shape(tmp_path):
    pin_paths = []
    for name, pin in (
        ("linux.json", _pin()),
        (
            "macos.json",
            _pin(
                operating_system="macos",
                architecture="arm64",
                url="https://baseten-public.s3.us-west-2.amazonaws.com/bin/cannery/1.2.3/cannery-1.2.3-macos-arm64",
                sha256="b" * 64,
            ),
        ),
    ):
        path = tmp_path / name
        path.write_text(json.dumps(pin))
        pin_paths.append(path)
    output = tmp_path / "bundled.json"

    subprocess.run([sys.executable, SCRIPT, *pin_paths, "--output", output], check=True)

    assert json.loads(output.read_text()) == {
        "artifacts": [
            _pin(),
            _pin(
                operating_system="macos",
                architecture="arm64",
                url="https://baseten-public.s3.us-west-2.amazonaws.com/bin/cannery/1.2.3/cannery-1.2.3-macos-arm64",
                sha256="b" * 64,
            ),
        ]
    }


def test_release_pin_rejects_manifest_only_fields(tmp_path):
    pin_path = tmp_path / "manifest.json"
    pin_path.write_text(json.dumps({**_pin(), "signed": False}))

    result = subprocess.run(
        [sys.executable, SCRIPT, pin_path, "--output", tmp_path / "output.json"],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "must contain exactly" in result.stderr
    assert not (tmp_path / "output.json").exists()


def test_checked_in_artifact_table_is_valid_and_empty():
    subprocess.run(
        [sys.executable, SCRIPT, "--check", "--output", BUNDLED_PINS], check=True
    )

    assert json.loads(BUNDLED_PINS.read_text()) == {"artifacts": []}
