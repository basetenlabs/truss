import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
from google.protobuf import json_format

from truss.cli.cannery.generated import cannery_cli_v1_pb2

GENERATED_ROOT = Path(__file__).parents[3] / "cli" / "cannery" / "generated"
FIXTURES_ROOT = GENERATED_ROOT / "fixtures" / "protojson"
REPO_ROOT = Path(__file__).parents[4]


def test_vendored_contract_matches_hash_manifest():
    manifest = json.loads((GENERATED_ROOT / "cannery_cli_v1.manifest.json").read_text())

    for relative_path, expected_hash in manifest["contract_sha256"].items():
        contents = (GENERATED_ROOT / relative_path).read_bytes()
        assert hashlib.sha256(contents).hexdigest() == expected_hash


def test_vendored_fixture_set_matches_protocol_contract():
    assert {path.name for path in FIXTURES_ROOT.glob("*.ndjson")} == {
        "list-quota-error.ndjson",
        "list-success.ndjson",
        "pull-cancelled.ndjson",
        "pull-integrity-error.ndjson",
        "pull-invalid-include.ndjson",
        "pull-no-match.ndjson",
        "pull-success.ndjson",
        "push-success.ndjson",
        "push-throttled.ndjson",
        "show-not-found.ndjson",
        "show-success.ndjson",
    }


@pytest.mark.parametrize("fixture_path", sorted(FIXTURES_ROOT.glob("*.ndjson")))
def test_cross_repo_protojson_golden_fixture(fixture_path):
    records = []
    for line in fixture_path.read_text().splitlines():
        records.append(json_format.Parse(line, cannery_cli_v1_pb2.MachineRecordV1()))

    assert records[0].WhichOneof("payload") == "started"
    assert records[-1].WhichOneof("payload") in {"result", "error", "cancelled"}
    assert [record.sequence for record in records] == list(range(1, len(records) + 1))


def test_vendored_generated_contract_has_no_drift():
    subprocess.run(
        [
            sys.executable,
            REPO_ROOT / "scripts" / "sync_cannery_protocol_v1.py",
            "--check",
        ],
        check=True,
    )
