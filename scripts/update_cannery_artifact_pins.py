#!/usr/bin/env python3
"""Import reviewed trusted producer release pins into the Truss trust table."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
from urllib.parse import urlparse

PIN_FIELDS = {
    "cannery_version",
    "protocol_version",
    "operating_system",
    "architecture",
    "url",
    "size_bytes",
    "sha256",
}
SUPPORTED_PLATFORMS = {
    ("linux", "arm64"),
    ("linux", "x86_64"),
    ("macos", "arm64"),
    ("macos", "x86_64"),
}
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "truss" / "cli" / "cannery" / "bundled_artifacts.json"


def _load_pin(path: Path) -> Dict[str, Any]:
    try:
        pin = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: could not read a JSON release pin") from exc
    if not isinstance(pin, dict) or set(pin) != PIN_FIELDS:
        raise ValueError(
            f"{path}: release pin must contain exactly {sorted(PIN_FIELDS)}"
        )
    _validate_pin(pin, path)
    return pin


def _validate_pin(pin: Dict[str, Any], path: Path) -> None:
    version = pin["cannery_version"]
    if not isinstance(version, str) or not version:
        raise ValueError(f"{path}: cannery_version must be a non-empty string")
    if type(pin["protocol_version"]) is not int or pin["protocol_version"] != 1:
        raise ValueError(f"{path}: protocol_version must be 1")
    platform = (pin["operating_system"], pin["architecture"])
    if platform not in SUPPORTED_PLATFORMS:
        raise ValueError(f"{path}: unsupported platform {platform}")
    if urlparse(pin["url"]).scheme.lower() != "https":
        raise ValueError(f"{path}: url must use HTTPS")
    if type(pin["size_bytes"]) is not int or pin["size_bytes"] <= 0:
        raise ValueError(f"{path}: size_bytes must be a positive integer")
    digest = pin["sha256"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(f"{path}: sha256 must be 64 lowercase hexadecimal digits")


def _build_table(pins: Iterable[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    artifacts = sorted(
        pins, key=lambda pin: (pin["operating_system"], pin["architecture"])
    )
    platforms: set[Tuple[str, str]] = set()
    versions = set()
    for pin in artifacts:
        platform = (pin["operating_system"], pin["architecture"])
        if platform in platforms:
            raise ValueError(f"duplicate release pin for {platform}")
        platforms.add(platform)
        versions.add(pin["cannery_version"])
    if len(versions) > 1:
        raise ValueError("all release pins must use the same cannery_version")
    return {"artifacts": artifacts}


def _write_table(path: Path, table: Dict[str, List[Dict[str, Any]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(table, output, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temp_path, path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def _check_table(path: Path) -> None:
    try:
        table = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: could not read the bundled artifact table") from exc
    if not isinstance(table, dict) or set(table) != {"artifacts"}:
        raise ValueError(f"{path}: table must contain exactly an artifacts array")
    artifacts = table["artifacts"]
    if not isinstance(artifacts, list):
        raise ValueError(f"{path}: artifacts must be an array")
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise ValueError(f"{path}: each artifact pin must be an object")
        if set(artifact) != PIN_FIELDS:
            raise ValueError(f"{path}: bundled pin fields do not match release pins")
        _validate_pin(artifact, path)
    if _build_table(artifacts) != table:
        raise ValueError(f"{path}: artifact pins are not canonical")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "pins",
        nargs="*",
        type=Path,
        help="Trusted producer *.truss-pin.json release manifests.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check", action="store_true", help="Validate the checked-in table only."
    )
    args = parser.parse_args()
    try:
        if args.check:
            if args.pins:
                parser.error("--check does not accept release pin paths")
            _check_table(args.output)
            print("Bundled Cannery artifact pins are valid.")
            return 0
        if not args.pins:
            parser.error(
                "at least one trusted producer *.truss-pin.json manifest is required"
            )
        table = _build_table(_load_pin(path) for path in args.pins)
        _write_table(args.output, table)
    except ValueError as exc:
        parser.error(str(exc))
    print(f"Updated bundled Cannery artifact pins in {args.output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
