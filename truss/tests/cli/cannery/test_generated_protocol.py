import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
from google.protobuf import json_format

from truss.cli.cannery.generated import cannery_cli_v1_pb2

GENERATED_ROOT = Path(__file__).parents[3] / "cli" / "cannery" / "generated"
PROTOJSON_FIXTURES = GENERATED_ROOT / "fixtures" / "protojson"
BINARY_FIXTURES = GENERATED_ROOT / "fixtures" / "protobuf-delimited"
REPO_ROOT = Path(__file__).parents[4]
EXPECTED_FIXTURE_STEMS = {
    "list-quota-error",
    "list-success",
    "pull-cancelled",
    "pull-integrity-error",
    "pull-invalid-include",
    "pull-no-match",
    "pull-success",
    "push-success",
    "push-throttled",
    "show-not-found",
    "show-success",
}


def _encode_varint(value):
    encoded = bytearray()
    while value >= 0x80:
        encoded.append((value & 0x7F) | 0x80)
        value >>= 7
    encoded.append(value)
    return bytes(encoded)


def _decode_binary_fixture(stream):
    records = []
    framed_records = []
    offset = 0
    while offset < len(stream):
        frame_start = offset
        payload_size = 0
        shift = 0
        while True:
            byte = stream[offset]
            offset += 1
            payload_size |= (byte & 0x7F) << shift
            if not byte & 0x80:
                break
            shift += 7
        payload = stream[offset : offset + payload_size]
        offset += payload_size
        record = cannery_cli_v1_pb2.MachineRecordV1()
        record.ParseFromString(payload)
        records.append(record)
        framed_records.append(stream[frame_start:offset])
    return records, framed_records


def test_vendored_contract_matches_hash_manifest():
    manifest = json.loads((GENERATED_ROOT / "cannery_cli_v1.manifest.json").read_text())

    for relative_path, expected_hash in manifest["contract_sha256"].items():
        contents = (GENERATED_ROOT / relative_path).read_bytes()
        assert hashlib.sha256(contents).hexdigest() == expected_hash


def test_vendored_fixture_set_matches_protocol_contract():
    assert {path.stem for path in PROTOJSON_FIXTURES.glob("*.ndjson")} == (
        EXPECTED_FIXTURE_STEMS
    )
    assert {path.stem for path in BINARY_FIXTURES.glob("*.bin")} == (
        EXPECTED_FIXTURE_STEMS
    )


def test_pull_request_field_numbers_remain_compatible():
    fields = cannery_cli_v1_pb2.PullRequestV1.DESCRIPTOR.fields_by_name

    assert fields["include_paths"].number == 5
    assert fields["restart"].number == 6


@pytest.mark.parametrize("fixture_path", sorted(BINARY_FIXTURES.glob("*.bin")))
def test_cross_repo_binary_golden_matches_rust_fixture_and_decoded_review_artifact(
    fixture_path,
):
    stream = fixture_path.read_bytes()
    records, framed_records = _decode_binary_fixture(stream)
    decoded_records = [
        json_format.Parse(line, cannery_cli_v1_pb2.MachineRecordV1())
        for line in (PROTOJSON_FIXTURES / f"{fixture_path.stem}.ndjson")
        .read_text()
        .splitlines()
    ]

    assert records[0].WhichOneof("payload") == "started"
    assert records[-1].WhichOneof("payload") in {"result", "error", "cancelled"}
    assert [record.sequence for record in records] == list(range(1, len(records) + 1))
    assert records == decoded_records
    assert (
        b"".join(
            _encode_varint(len(payload)) + payload
            for payload in (
                record.SerializeToString(deterministic=True) for record in records
            )
        )
        == stream
    )
    assert b"".join(framed_records) == stream


def test_vendored_generated_contract_has_no_drift():
    subprocess.run(
        [
            sys.executable,
            REPO_ROOT / "scripts" / "sync_cannery_protocol_v1.py",
            "--check",
        ],
        check=True,
    )
