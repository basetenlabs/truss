#!/usr/bin/env python3
"""Synchronize and verify the vendored Cannery machine protocol v1 contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterable, Optional

from grpc_tools import protoc

GENERATOR = "grpcio-tools==1.73.1"
CONTRACT_FILES = (
    "cannery_cli_v1.proto",
    "v1.md",
    "fixtures/protojson/list-quota-error.ndjson",
    "fixtures/protojson/list-success.ndjson",
    "fixtures/protojson/pull-cancelled.ndjson",
    "fixtures/protojson/pull-integrity-error.ndjson",
    "fixtures/protojson/pull-success.ndjson",
    "fixtures/protojson/push-success.ndjson",
    "fixtures/protojson/push-throttled.ndjson",
    "fixtures/protojson/show-not-found.ndjson",
    "fixtures/protojson/show-success.ndjson",
)
GENERATED_FILES = ("cannery_cli_v1_pb2.py", "cannery_cli_v1_pb2.pyi")

REPO_ROOT = Path(__file__).resolve().parents[1]
VENDORED_ROOT = REPO_ROOT / "truss" / "cli" / "cannery" / "generated"
MANIFEST_PATH = VENDORED_ROOT / "cannery_cli_v1.manifest.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hashes(root: Path, files: Iterable[str]) -> Dict[str, str]:
    return {relative: _sha256(root / relative) for relative in files}


def _generate(root: Path) -> None:
    result = protoc.main(
        [
            "grpc_tools.protoc",
            f"-I{root}",
            f"--python_out={root}",
            f"--pyi_out={root}",
            str(root / "cannery_cli_v1.proto"),
        ]
    )
    if result:
        raise RuntimeError(f"protoc exited with status {result}")


def _write_manifest(root: Path) -> None:
    payload = {"generator": GENERATOR, "contract_sha256": _hashes(root, CONTRACT_FILES)}
    (root / MANIFEST_PATH.name).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _copy_contract(source_root: Path, destination_root: Path) -> None:
    for relative in CONTRACT_FILES:
        source = source_root / relative
        if not source.is_file():
            raise FileNotFoundError(f"missing Cannery contract file: {source}")
        destination = destination_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)


def _build_expected(source_root: Path, destination_root: Path) -> None:
    _copy_contract(source_root, destination_root)
    _generate(destination_root)
    _write_manifest(destination_root)


def _check_file(expected: Path, actual: Path) -> Optional[str]:
    if not actual.is_file():
        return f"missing vendored file: {actual.relative_to(REPO_ROOT)}"
    if expected.read_bytes() != actual.read_bytes():
        return f"out-of-date vendored file: {actual.relative_to(REPO_ROOT)}"
    return None


def _check(source_root: Optional[Path]) -> int:
    contract_root = source_root or VENDORED_ROOT
    with tempfile.TemporaryDirectory(prefix="cannery-protocol-v1-") as temp_dir:
        expected_root = Path(temp_dir)
        _build_expected(contract_root, expected_root)
        failures = [
            failure
            for relative in (*CONTRACT_FILES, *GENERATED_FILES, MANIFEST_PATH.name)
            if (
                failure := _check_file(
                    expected_root / relative, VENDORED_ROOT / relative
                )
            )
        ]
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    print("Cannery protocol v1 generated files are in sync.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        help="Directory containing the canonical cannery_cli_v1.proto and fixtures.",
    )
    parser.add_argument(
        "--check", action="store_true", help="Verify without modifying vendored files."
    )
    args = parser.parse_args()

    source_root = args.source_root.resolve() if args.source_root else None
    if args.check:
        return _check(source_root)
    if source_root is None:
        parser.error("--source-root is required unless --check is used")

    VENDORED_ROOT.mkdir(parents=True, exist_ok=True)
    _build_expected(source_root, VENDORED_ROOT)
    print(f"Synchronized Cannery protocol v1 from {source_root}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
