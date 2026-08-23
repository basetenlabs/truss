import io
import json
from pathlib import Path

import pytest

from truss.cli.cannery.errors import CanneryProtocolError
from truss.cli.cannery.protocol import (
    Phase0ProtocolConsumer,
    V1ProtocolConsumer,
    parse_protocol_bootstrap,
)

FIXTURES = (
    Path(__file__).parents[3]
    / "cli"
    / "cannery"
    / "generated"
    / "fixtures"
    / "protojson"
)


def test_phase_zero_consumer_retains_bounded_progress_state():
    def progress_lines():
        for completed in range(10_000):
            yield (
                json.dumps(
                    {
                        "protocol_version": 1,
                        "type": "progress",
                        "operation": "push",
                        "phase": "upload",
                        "bytes_done": completed,
                    }
                )
                + "\n"
            )

    session = Phase0ProtocolConsumer().start(
        io.StringIO('{"protocol_version":1}'), progress_lines(), lambda _message: None
    )

    assert session.read_result() == {"protocol_version": 1}
    session.finish(0)
    assert session.last_phase == "upload"
    assert session.terminal_error is None
    assert not any(isinstance(value, list) for value in vars(session).values())


def _fixture(name):
    return (FIXTURES / name).read_text()


def _consume(stream, command, return_code):
    rendered = []
    session = V1ProtocolConsumer(command).start(
        io.StringIO(stream), io.StringIO(""), rendered.append
    )
    result = session.read_result()
    session.finish(return_code)
    return session, result, rendered


@pytest.mark.parametrize(
    ("fixture_name", "command", "expected_field"),
    [
        ("push-success.ndjson", "push", "manifest_digest"),
        ("list-success.ndjson", "ls", "references"),
        ("show-success.ndjson", "show", "file_page"),
        ("pull-success.ndjson", "pull", "content_verified"),
    ],
)
def test_v1_consumer_converts_typed_success_results(
    fixture_name, command, expected_field
):
    session, result, rendered = _consume(_fixture(fixture_name), command, 0)

    assert expected_field in result
    assert session.terminal_error is None
    assert rendered


@pytest.mark.parametrize(
    ("fixture_name", "command", "category"),
    [
        ("push-throttled.ndjson", "push", "throttled"),
        ("list-quota-error.ndjson", "ls", "quota"),
        ("show-not-found.ndjson", "show", "not_found"),
        ("pull-integrity-error.ndjson", "pull", "integrity"),
    ],
)
def test_v1_consumer_maps_typed_errors(fixture_name, command, category):
    session, result, _ = _consume(_fixture(fixture_name), command, 1)

    assert result == {}
    assert session.terminal_error["category"] == category
    assert session.terminal_error["details"]


def test_v1_consumer_preserves_retry_after_milliseconds():
    session, _, _ = _consume(_fixture("push-throttled.ndjson"), "push", 1)

    assert session.terminal_error["retry_after_ms"] == 30_000
    assert session.terminal_error["retryable"] is True


def test_v1_consumer_maps_cancelled_exit_130():
    session, result, _ = _consume(_fixture("pull-cancelled.ndjson"), "pull", 130)

    assert result == {}
    assert session.cancelled


@pytest.mark.parametrize(
    "document",
    [
        "",
        "{}\n",
        '{"bootstrap_version":2,"cannery_version":"1","supported_machine_protocols":[1],"supported_encodings":["protojson-ndjson"]}\n',
        '{"bootstrap_version":1,"cannery_version":"1","supported_machine_protocols":[2],"supported_encodings":["protojson-ndjson"]}\n',
        '{"bootstrap_version":1,"cannery_version":"1","supported_machine_protocols":[1],"supported_encodings":["protobuf-delimited"]}\n',
    ],
)
def test_protocol_bootstrap_mismatch_is_rejected(document):
    with pytest.raises(CanneryProtocolError):
        parse_protocol_bootstrap(document, 0)


def test_protocol_bootstrap_accepts_v1_protojson():
    bootstrap = parse_protocol_bootstrap(
        '{"bootstrap_version":1,"cannery_version":"1.2.3",'
        '"supported_machine_protocols":[1],'
        '"supported_encodings":["protojson-ndjson"]}\n',
        0,
    )

    assert bootstrap.cannery_version == "1.2.3"


def _replace_record(stream, index, update):
    lines = stream.splitlines()
    document = json.loads(lines[index])
    update(document)
    lines[index] = json.dumps(document, separators=(",", ":"))
    return "\n".join(lines) + "\n"


@pytest.mark.parametrize(
    ("update", "message"),
    [
        (lambda record: record.update(sequence="99"), "sequence"),
        (lambda record: record.update(operationId="different"), "operationId"),
        (lambda record: record.update(operation="OPERATION_PULL"), "operation"),
        (lambda record: record.update(unknownField=True), "does not match v1"),
        (lambda record: record.update(operation=1), "symbolic enum"),
    ],
)
def test_v1_consumer_rejects_sequence_metadata_and_unknown_values(update, message):
    stream = _replace_record(_fixture("push-success.ndjson"), 1, update)

    with pytest.raises(CanneryProtocolError, match=message):
        _consume(stream, "push", 0)


def test_v1_consumer_requires_exactly_one_terminal_record():
    lines = _fixture("push-success.ndjson").splitlines()

    with pytest.raises(CanneryProtocolError, match="without a terminal"):
        _consume("\n".join(lines[:-1]) + "\n", "push", 0)

    with pytest.raises(CanneryProtocolError, match="after its terminal"):
        _consume("\n".join([*lines, lines[-1]]) + "\n", "push", 0)


def test_v1_read_result_returns_at_terminal_then_finish_drains_eof():
    lines = iter(_fixture("push-success.ndjson").splitlines(keepends=True))

    class TerminalThenPausedStdout:
        allow_eof = False

        def readline(self, _size):
            try:
                return next(lines)
            except StopIteration:
                if not self.allow_eof:
                    raise AssertionError("read_result read beyond the terminal record")
                return ""

    stdout = TerminalThenPausedStdout()
    session = V1ProtocolConsumer("push").start(
        stdout, io.StringIO(""), lambda _message: None
    )

    result = session.read_result()

    assert result["manifest_digest"].startswith("b3:")
    assert not session._stream_complete
    stdout.allow_eof = True
    session.finish(0)
    assert session._stream_complete


def test_v1_consumer_rejects_exit_terminal_mismatch():
    session = V1ProtocolConsumer("push").start(
        io.StringIO(_fixture("push-success.ndjson")),
        io.StringIO(""),
        lambda _message: None,
    )
    session.read_result()

    with pytest.raises(CanneryProtocolError, match="exit status"):
        session.finish(1)


def test_v1_consumer_retains_bounded_progress_state():
    operation_id = "bounded-progress"
    started = {
        "protocolVersion": 1,
        "sequence": "1",
        "operationId": operation_id,
        "operation": "OPERATION_PUSH",
        "started": {
            "request": {
                "protocolVersion": 1,
                "operationId": operation_id,
                "push": {"localPath": ".", "reference": "bdn://dev/model"},
            },
            "canneryVersion": "1.2.3",
        },
    }
    result = {
        "protocolVersion": 1,
        "sequence": "10002",
        "operationId": operation_id,
        "operation": "OPERATION_PUSH",
        "result": {
            "push": {
                "manifestDigest": "b3:test",
                "canonicalReference": "bdn://dev/model",
            }
        },
    }
    lines = [json.dumps(started)]
    for sequence in range(2, 10_002):
        lines.append(
            json.dumps(
                {
                    "protocolVersion": 1,
                    "sequence": str(sequence),
                    "operationId": operation_id,
                    "operation": "OPERATION_PUSH",
                    "progress": {
                        "phase": "upload",
                        "itemsDone": str(sequence),
                        "elapsedSeconds": sequence / 100,
                    },
                }
            )
        )
    lines.append(json.dumps(result))

    session, parsed, _ = _consume("\n".join(lines) + "\n", "push", 0)

    assert parsed["manifest_digest"] == "b3:test"
    assert len(session._progress_counters) == 3
    assert not any(isinstance(value, list) for value in vars(session).values())


def test_v1_consumer_enforces_bounded_result_pages():
    stream = _fixture("show-success.ndjson")

    def add_files(record):
        file_entry = {"path": "file", "kind": "FILE_ENTRY_KIND_FILE", "sizeBytes": "1"}
        record["result"]["show"]["filePage"]["files"] = [file_entry] * 1_001

    stream = _replace_record(stream, 2, add_files)

    with pytest.raises(CanneryProtocolError, match="page limit"):
        _consume(stream, "show", 0)


def test_v1_consumer_bounds_captured_stderr():
    session = V1ProtocolConsumer("push").start(
        io.StringIO(_fixture("push-success.ndjson")),
        io.StringIO("x" * 100_000),
        lambda _message: None,
    )
    session.read_result()
    session.finish(0)

    assert len(session.stderr_diagnostic) < 66_000
    assert "truncated" in session.stderr_diagnostic
