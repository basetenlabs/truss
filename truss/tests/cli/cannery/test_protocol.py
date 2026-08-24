import io
import json
from pathlib import Path

import pytest
from google.protobuf import json_format

from truss.cli.cannery import v1_protocol
from truss.cli.cannery.errors import CanneryProtocolError
from truss.cli.cannery.generated import cannery_cli_v1_pb2 as protocol_v1
from truss.cli.cannery.protocol import (
    Phase0ProtocolConsumer,
    V1ProtocolConsumer,
    V1ProtoJSONProtocolConsumer,
    parse_protocol_bootstrap,
)

GENERATED_ROOT = Path(__file__).parents[3] / "cli" / "cannery" / "generated"
BINARY_FIXTURES = GENERATED_ROOT / "fixtures" / "protobuf-delimited"
PROTOJSON_FIXTURES = GENERATED_ROOT / "fixtures" / "protojson"


def _binary_fixture(name):
    return (BINARY_FIXTURES / name.replace(".ndjson", ".bin")).read_bytes()


def _protojson_fixture(name):
    return (PROTOJSON_FIXTURES / name.replace(".bin", ".ndjson")).read_bytes()


def _encode_varint(value):
    encoded = bytearray()
    while value >= 0x80:
        encoded.append((value & 0x7F) | 0x80)
        value >>= 7
    encoded.append(value)
    return bytes(encoded)


def _frame_payload(payload):
    return _encode_varint(len(payload)) + payload


def _frame_record(record):
    return _frame_payload(record.SerializeToString(deterministic=True))


def _split_payloads(stream):
    payloads = []
    offset = 0
    while offset < len(stream):
        payload_size = 0
        shift = 0
        while True:
            byte = stream[offset]
            offset += 1
            payload_size |= (byte & 0x7F) << shift
            if not byte & 0x80:
                break
            shift += 7
        payloads.append(stream[offset : offset + payload_size])
        offset += payload_size
    return payloads


def _decode_records(stream):
    records = []
    for payload in _split_payloads(stream):
        record = protocol_v1.MachineRecordV1()
        record.ParseFromString(payload)
        records.append(record)
    return records


def _encode_records(records):
    return b"".join(_frame_record(record) for record in records)


def _replace_record(stream, index, update):
    records = _decode_records(stream)
    update(records[index])
    return _encode_records(records)


def _replace_protojson_record(stream, index, update):
    lines = stream.decode().splitlines()
    document = json.loads(lines[index])
    update(document)
    lines[index] = json.dumps(document, separators=(",", ":"))
    return ("\n".join(lines) + "\n").encode()


def _consume(stream, command, return_code):
    rendered = []
    session = V1ProtocolConsumer(command).start(
        io.BytesIO(stream), io.BytesIO(), rendered.append
    )
    result = session.read_result()
    session.finish(return_code)
    return session, result, rendered


def _consume_protojson(stream, command, return_code):
    rendered = []
    session = V1ProtoJSONProtocolConsumer(command).start(
        io.BytesIO(stream), io.BytesIO(), rendered.append
    )
    result = session.read_result()
    session.finish(return_code)
    return session, result, rendered


class _OneByteReads(io.BytesIO):
    def read(self, size=-1):
        if size < 0:
            size = 1
        return super().read(min(size, 1))


def test_phase_zero_consumer_retains_bounded_progress_state():
    progress = bytearray()
    for completed in range(10_000):
        progress.extend(
            json.dumps(
                {
                    "protocol_version": 1,
                    "type": "progress",
                    "operation": "push",
                    "phase": "upload",
                    "bytes_done": completed,
                }
            ).encode()
            + b"\n"
        )

    session = Phase0ProtocolConsumer().start(
        io.BytesIO(b'{"protocol_version":1}'),
        io.BytesIO(progress),
        lambda _message: None,
    )

    assert session.read_result() == {"protocol_version": 1}
    session.finish(0)
    assert session.last_phase == "upload"
    assert session.terminal_error is None
    assert not any(isinstance(value, list) for value in vars(session).values())


@pytest.mark.parametrize(
    ("fixture_name", "command", "expected_field"),
    [
        ("push-success.bin", "push", "manifest_digest"),
        ("list-success.bin", "ls", "references"),
        ("show-success.bin", "show", "file_page"),
        ("pull-success.bin", "pull", "content_verified"),
    ],
)
def test_v1_consumer_converts_binary_success_results(
    fixture_name, command, expected_field
):
    session, result, rendered = _consume(_binary_fixture(fixture_name), command, 0)

    assert expected_field in result
    assert session.terminal_error is None
    assert rendered


@pytest.mark.parametrize(
    ("fixture_name", "command", "category"),
    [
        ("push-throttled.bin", "push", "throttled"),
        ("list-quota-error.bin", "ls", "quota"),
        ("show-not-found.bin", "show", "not_found"),
        ("pull-integrity-error.bin", "pull", "integrity"),
    ],
)
def test_v1_consumer_maps_binary_typed_errors(fixture_name, command, category):
    session, result, _ = _consume(_binary_fixture(fixture_name), command, 1)

    assert result == {}
    assert session.terminal_error["category"] == category
    assert session.terminal_error["details"]


def test_v1_consumer_preserves_retry_after_milliseconds():
    session, _, _ = _consume(_binary_fixture("push-throttled.bin"), "push", 1)

    assert session.terminal_error["retry_after_ms"] == 30_000
    assert session.terminal_error["retryable"] is True


def test_v1_pull_preserves_selected_and_optional_volume_totals():
    _, result, rendered = _consume(_binary_fixture("pull-success.bin"), "pull", 0)

    assert result["logical_bytes"] == "268435456"
    assert result["file_count"] == "3"
    assert result["directory_count"] == "1"
    assert result["volume_logical_bytes"] == "1073741824"
    assert result["volume_file_count"] == "12"
    assert result["volume_directory_count"] == "3"
    assert "1/3 files" in rendered[0]
    assert "67108864/268435456 bytes" in rendered[0]
    assert all("/12 files" not in message for message in rendered)


@pytest.mark.parametrize(
    ("camel_name", "snake_name"),
    [
        ("logicalBytes", "logical_bytes"),
        ("fileCount", "file_count"),
        ("directoryCount", "directory_count"),
    ],
)
def test_explicit_protojson_debug_consumer_requires_selected_totals(
    camel_name, snake_name
):
    def omit_selected_total(record):
        pull = record["result"]["pull"]
        pull.pop(camel_name, None)
        pull.pop(snake_name, None)

    stream = _replace_protojson_record(
        _protojson_fixture("pull-success.ndjson"), 3, omit_selected_total
    )

    with pytest.raises(CanneryProtocolError, match="required selected totals"):
        _consume_protojson(stream, "pull", 0)


def test_v1_consumer_maps_cancelled_exit_130():
    session, result, _ = _consume(_binary_fixture("pull-cancelled.bin"), "pull", 130)

    assert result == {}
    assert session.cancelled


@pytest.mark.parametrize(
    "document",
    [
        "",
        "{}\n",
        '{"bootstrap_version":2,"cannery_version":"1","supported_machine_protocols":[1],"supported_encodings":["protobuf-delimited"]}\n',
        '{"bootstrap_version":1,"cannery_version":"1","supported_machine_protocols":[2],"supported_encodings":["protobuf-delimited"]}\n',
        '{"bootstrap_version":1,"cannery_version":"1","supported_machine_protocols":[1],"supported_encodings":["protojson-ndjson"]}\n',
    ],
)
def test_protocol_bootstrap_mismatch_is_rejected(document):
    with pytest.raises(CanneryProtocolError):
        parse_protocol_bootstrap(document, 0)


def test_protocol_bootstrap_requires_binary_and_accepts_optional_debug_encoding():
    bootstrap = parse_protocol_bootstrap(
        '{"bootstrap_version":1,"cannery_version":"1.2.3",'
        '"supported_machine_protocols":[1],'
        '"supported_encodings":["protobuf-delimited","protojson-ndjson"]}\n',
        0,
    )

    assert bootstrap.cannery_version == "1.2.3"


def test_v1_pull_accepts_restart_request_without_changing_result():
    def enable_restart(record):
        record.started.request.pull.restart = True

    stream = _replace_record(_binary_fixture("pull-success.bin"), 0, enable_restart)

    _, result, _ = _consume(stream, "pull", 0)

    assert result["content_verified"] is True
    assert result["manifest_digest"].startswith("b3:")


def test_explicit_protojson_debug_consumer_rejects_non_boolean_restart():
    def set_invalid_restart(record):
        record["started"]["request"]["pull"]["restart"] = "yes"

    stream = _replace_protojson_record(
        _protojson_fixture("pull-success.ndjson"), 0, set_invalid_restart
    )

    with pytest.raises(CanneryProtocolError, match="does not match v1"):
        _consume_protojson(stream, "pull", 0)


def test_v1_pull_preserves_reused_bytes():
    def add_resumed_transfer_counts(record):
        record.result.pull.downloaded_bytes = 201_326_592
        record.result.pull.reused_bytes = 67_108_864

    stream = _replace_record(
        _binary_fixture("pull-success.bin"), 3, add_resumed_transfer_counts
    )

    _, result, _ = _consume(stream, "pull", 0)

    assert result["downloaded_bytes"] == "201326592"
    assert result["reused_bytes"] == "67108864"


@pytest.mark.parametrize(
    ("update", "message"),
    [
        (lambda record: setattr(record, "sequence", 99), "sequence"),
        (lambda record: setattr(record, "operation_id", "different"), "operationId"),
        (
            lambda record: setattr(record, "operation", protocol_v1.OPERATION_PULL),
            "operation",
        ),
        (lambda record: setattr(record, "operation", 99), "enum value"),
    ],
)
def test_v1_consumer_rejects_binary_sequence_metadata_and_operation_values(
    update, message
):
    stream = _replace_record(_binary_fixture("push-success.bin"), 1, update)

    with pytest.raises(CanneryProtocolError, match=message):
        _consume(stream, "push", 0)


def test_binary_additive_unknown_fields_are_ignored():
    payloads = _split_payloads(_binary_fixture("push-success.bin"))
    unknown_varint_field = _encode_varint((1_000 << 3) | 0) + _encode_varint(7)
    payloads[1] += unknown_varint_field
    stream = b"".join(_frame_payload(payload) for payload in payloads)

    _, result, _ = _consume(stream, "push", 0)

    assert result["manifest_digest"].startswith("b3:")


@pytest.mark.parametrize(
    ("fixture_name", "command", "record_index", "update", "message"),
    [
        (
            "list-success.bin",
            "ls",
            2,
            lambda record: setattr(
                record.result.list.references.references[0], "kind", 99
            ),
            "reference kind",
        ),
        (
            "show-success.bin",
            "show",
            2,
            lambda record: setattr(record.result.show.file_page.files[0], "kind", 99),
            "file kind",
        ),
        (
            "show-not-found.bin",
            "show",
            2,
            lambda record: setattr(record.error, "category", 99),
            "unsupported category",
        ),
    ],
)
def test_unknown_required_binary_enum_numbers_fail_semantics(
    fixture_name, command, record_index, update, message
):
    stream = _replace_record(_binary_fixture(fixture_name), record_index, update)

    with pytest.raises(CanneryProtocolError, match=message):
        _consume(stream, command, 1 if "not-found" in fixture_name else 0)


@pytest.mark.parametrize(
    ("update", "message"),
    [
        (
            lambda records: records[0].started.request.ClearField("push"),
            "command variant",
        ),
        (lambda records: records[1].ClearField("progress"), "missing its payload"),
        (lambda records: records[-1].result.ClearField("push"), "operation variant"),
    ],
)
def test_required_binary_oneofs_are_validated(update, message):
    records = _decode_records(_binary_fixture("push-success.bin"))
    update(records)

    with pytest.raises(CanneryProtocolError, match=message):
        _consume(_encode_records(records), "push", 0)


def test_required_binary_list_page_oneof_is_validated():
    def clear_page(record):
        record.result.list.ClearField("references")

    stream = _replace_record(_binary_fixture("list-success.bin"), 2, clear_page)

    with pytest.raises(CanneryProtocolError, match="missing its page"):
        _consume(stream, "ls", 0)


def test_v1_consumer_requires_exactly_one_terminal_record_and_clean_eof():
    payloads = _split_payloads(_binary_fixture("push-success.bin"))
    missing_terminal = b"".join(_frame_payload(payload) for payload in payloads[:-1])
    duplicate_terminal = _binary_fixture("push-success.bin") + _frame_payload(
        payloads[-1]
    )

    with pytest.raises(CanneryProtocolError, match="without a terminal"):
        _consume(missing_terminal, "push", 0)

    with pytest.raises(CanneryProtocolError, match="after its terminal"):
        _consume(duplicate_terminal, "push", 0)


@pytest.mark.parametrize("trailing", [b"\x80", b"\x05\x08"])
def test_partial_trailing_binary_bytes_after_terminal_are_protocol_errors(trailing):
    with pytest.raises(CanneryProtocolError, match="truncated"):
        _consume(_binary_fixture("push-success.bin") + trailing, "push", 0)


def test_v1_read_result_returns_at_terminal_then_finish_reads_immediate_eof():
    class TerminalThenPausedStdout(io.BytesIO):
        allow_eof = False

        def read(self, size=-1):
            chunk = super().read(size)
            if not chunk and not self.allow_eof:
                raise AssertionError("read_result read beyond the terminal record")
            return chunk

    stdout = TerminalThenPausedStdout(_binary_fixture("push-success.bin"))
    session = V1ProtocolConsumer("push").start(
        stdout, io.BytesIO(), lambda _message: None
    )

    result = session.read_result()

    assert result["manifest_digest"].startswith("b3:")
    assert not session._stream_complete
    stdout.allow_eof = True
    session.finish(0)
    assert session._stream_complete


def test_v1_consumer_rejects_exit_terminal_mismatch():
    session = V1ProtocolConsumer("push").start(
        io.BytesIO(_binary_fixture("push-success.bin")),
        io.BytesIO(),
        lambda _message: None,
    )
    session.read_result()

    with pytest.raises(CanneryProtocolError, match="exit status"):
        session.finish(1)


def test_v1_consumer_handles_one_byte_partial_reads():
    rendered = []
    session = V1ProtocolConsumer("show").start(
        _OneByteReads(_binary_fixture("show-success.bin")),
        io.BytesIO(),
        rendered.append,
    )

    result = session.read_result()
    session.finish(0)

    assert result["manifest_digest"].startswith("b3:")
    assert rendered


def test_canonical_varint_reader_accepts_maximum_frame_boundary():
    maximum = v1_protocol._MAX_RECORD_BYTES
    stream = io.BytesIO(_encode_varint(maximum) + b"x" * maximum)

    assert v1_protocol._read_canonical_varint(stream, 1) == maximum
    assert len(v1_protocol._read_exact_payload(stream, maximum, 1)) == maximum


@pytest.mark.parametrize(
    ("prefix", "message"),
    [
        (b"\x00", "zero-length"),
        (b"\x80", "truncated"),
        (b"\x80\x00", "non-minimal"),
        (b"\x81\x80\x80\x00", "non-minimal"),
        (b"\x80\x80\x80\x80", "overlong"),
        (b"\x81\x80\x80\x04", "size limit"),
    ],
)
def test_canonical_varint_reader_rejects_malformed_prefixes(prefix, message):
    with pytest.raises(CanneryProtocolError, match=message):
        v1_protocol._DelimitedRecordReader(io.BytesIO(prefix)).read()


def test_oversize_prefix_is_rejected_before_body_read_or_allocation():
    local_path = b"/Users/customer/private/model"

    class ReadSpy(io.BytesIO):
        def __init__(self, value):
            super().__init__(value)
            self.read_sizes = []

        def read(self, size=-1):
            self.read_sizes.append(size)
            return super().read(size)

    stream = ReadSpy(_encode_varint(v1_protocol._MAX_RECORD_BYTES + 1) + local_path)

    with pytest.raises(CanneryProtocolError) as exc_info:
        v1_protocol._DelimitedRecordReader(stream).read()

    assert stream.read_sizes == [1, 1, 1, 1]
    assert local_path.decode() not in str(exc_info.value)


@pytest.mark.parametrize(
    ("stream", "message"),
    [
        (_frame_payload(b"\x0f"), "malformed Protobuf"),
        (_frame_payload(b"\x08\x00"), "default record"),
        (_encode_varint(4) + b"\x08\x01", "truncated payload"),
    ],
)
def test_malformed_default_and_truncated_binary_frames_fail(stream, message):
    with pytest.raises(CanneryProtocolError, match=message):
        _consume(stream, "push", 0)


def test_machine_payload_bytes_are_not_dumped_in_protocol_diagnostics():
    local_path = "/Users/customer/private/model"
    stream = _frame_payload(local_path.encode() + b"\xff")

    with pytest.raises(CanneryProtocolError) as exc_info:
        _consume(stream, "push", 0)

    assert local_path not in str(exc_info.value)
    assert repr(stream) not in str(exc_info.value)


def test_v1_consumer_rejects_clean_eof_before_terminal():
    with pytest.raises(CanneryProtocolError, match="without a terminal"):
        _consume(b"", "push", 0)

    records = _decode_records(_binary_fixture("push-success.bin"))
    with pytest.raises(CanneryProtocolError, match="without a terminal"):
        _consume(_encode_records(records[:1]), "push", 0)


def test_empty_stdout_exit_two_remains_a_usage_failure():
    session, result, _ = _consume(b"", "show", 2)

    assert result == {}
    assert session.terminal_error["category"] == "usage"


def test_v1_consumer_retains_bounded_progress_state():
    operation_id = "bounded-progress"
    records = [
        protocol_v1.MachineRecordV1(
            protocol_version=1,
            sequence=1,
            operation_id=operation_id,
            operation=protocol_v1.OPERATION_PUSH,
            started=protocol_v1.StartedV1(
                request=protocol_v1.CommandRequestV1(
                    protocol_version=1,
                    operation_id=operation_id,
                    push=protocol_v1.PushRequestV1(
                        local_path=".", reference="bdn://dev/model"
                    ),
                ),
                cannery_version="1.2.3",
            ),
        )
    ]
    for sequence in range(2, 10_002):
        records.append(
            protocol_v1.MachineRecordV1(
                protocol_version=1,
                sequence=sequence,
                operation_id=operation_id,
                operation=protocol_v1.OPERATION_PUSH,
                progress=protocol_v1.ProgressV1(
                    phase="upload", items_done=sequence, elapsed_seconds=sequence / 100
                ),
            )
        )
    records.append(
        protocol_v1.MachineRecordV1(
            protocol_version=1,
            sequence=10_002,
            operation_id=operation_id,
            operation=protocol_v1.OPERATION_PUSH,
            result=protocol_v1.ResultV1(
                push=protocol_v1.PushResultV1(
                    manifest_digest="b3:test", canonical_reference="bdn://dev/model"
                )
            ),
        )
    )

    session, parsed, _ = _consume(_encode_records(records), "push", 0)

    assert parsed["manifest_digest"] == "b3:test"
    assert len(session._progress_counters) == 3
    assert not any(isinstance(value, list) for value in vars(session).values())


def test_v1_consumer_enforces_bounded_result_pages():
    def add_files(record):
        for _ in range(1_001):
            record.result.show.file_page.files.add(
                path="file", kind=protocol_v1.FILE_ENTRY_KIND_FILE, size_bytes=1
            )

    stream = _replace_record(_binary_fixture("show-success.bin"), 2, add_files)

    with pytest.raises(CanneryProtocolError, match="page limit"):
        _consume(stream, "show", 0)


def test_v1_consumer_safely_decodes_bounds_and_redacts_stderr():
    secret = "bare-credential-value-9f4c"
    stderr = _OneByteReads(
        ("☃" + "x" * 100_000 + f"\nAuthorization: Bearer {secret}\n").encode()
    )
    session = V1ProtocolConsumer("push").start(
        io.BytesIO(_binary_fixture("push-success.bin")), stderr, lambda _message: None
    )
    session.read_result()
    session.finish(0)

    assert len(session.stderr_diagnostic) < 66_000
    assert "truncated" in session.stderr_diagnostic
    assert secret not in session.stderr_diagnostic
    assert "[REDACTED]" in session.stderr_diagnostic


def test_explicit_protojson_debug_consumer_parses_review_fixture():
    _, result, _ = _consume_protojson(
        _protojson_fixture("show-success.ndjson"), "show", 0
    )

    assert result["manifest_digest"].startswith("b3:")


def test_explicit_protojson_debug_consumer_rejects_unknown_fields_and_numeric_enums():
    unknown_field = _replace_protojson_record(
        _protojson_fixture("push-success.ndjson"),
        1,
        lambda record: record.update(unknownField=True),
    )
    numeric_enum = _replace_protojson_record(
        _protojson_fixture("push-success.ndjson"),
        1,
        lambda record: record.update(operation=1),
    )

    with pytest.raises(CanneryProtocolError, match="does not match v1"):
        _consume_protojson(unknown_field, "push", 0)
    with pytest.raises(CanneryProtocolError, match="symbolic enum"):
        _consume_protojson(numeric_enum, "push", 0)


def test_python_binary_encoding_matches_authoritative_fixture_records():
    for fixture_path in sorted(BINARY_FIXTURES.glob("*.bin")):
        stream = fixture_path.read_bytes()
        assert _encode_records(_decode_records(stream)) == stream
        decoded_path = PROTOJSON_FIXTURES / f"{fixture_path.stem}.ndjson"
        expected_records = [
            json_format.Parse(line, protocol_v1.MachineRecordV1())
            for line in decoded_path.read_text().splitlines()
        ]
        assert _decode_records(stream) == expected_records
