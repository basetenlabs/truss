import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from baseten_performance_client import PerformanceClient, RequestProcessingPreference

PARENT = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"

# Span export is opt-in per process; these tests cover propagation with export off.
pytestmark = pytest.mark.skipif(
    bool(os.environ.get("BASETEN_PERFORMANCE_CLIENT_OTLP_ENDPOINT")),
    reason="client span export is enabled in this environment",
)


class _EmbeddingsHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.traceparents.append(self.headers.get_all("traceparent") or [])
        self.server.tracestates.append(self.headers.get_all("tracestate") or [])
        payload = json.dumps(
            {
                "object": "list",
                "data": [
                    {"object": "embedding", "embedding": [0.1], "index": index}
                    for index, _ in enumerate(body["input"])
                ],
                "model": body["model"],
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):
        pass


@pytest.fixture
def embeddings_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _EmbeddingsHandler)
    server.traceparents = []
    server.tracestates = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()


def _client(server):
    return PerformanceClient(
        base_url=f"http://127.0.0.1:{server.server_address[1]}", api_key="test-key"
    )


def _sdk_tracer():
    # Only the ambient-context tests need the OpenTelemetry SDK.
    sdk_trace = pytest.importorskip("opentelemetry.sdk.trace")
    return sdk_trace.TracerProvider().get_tracer("test")


def test_traceparent_preference_round_trips():
    preference = RequestProcessingPreference(traceparent=PARENT)
    assert preference.traceparent == PARENT
    assert PARENT in repr(preference)
    preference.traceparent = None
    assert preference.traceparent is None
    assert RequestProcessingPreference().traceparent is None


def test_explicit_traceparent_is_forwarded(embeddings_server):
    _client(embeddings_server).embed(
        ["hello"],
        model="test-model",
        preference=RequestProcessingPreference(traceparent=PARENT),
    )
    assert embeddings_server.traceparents == [[PARENT]]


def test_no_parent_sends_no_traceparent(embeddings_server):
    _client(embeddings_server).embed(["hello"], model="test-model")
    assert embeddings_server.traceparents == [[]]


def test_invalid_traceparent_raises_value_error(embeddings_server):
    with pytest.raises(ValueError, match="traceparent"):
        _client(embeddings_server).embed(
            ["hello"],
            model="test-model",
            preference=RequestProcessingPreference(traceparent="not-a-traceparent"),
        )
    assert embeddings_server.traceparents == []


def test_active_opentelemetry_span_is_the_parent(embeddings_server):
    tracer = _sdk_tracer()
    with tracer.start_as_current_span("caller") as span:
        _client(embeddings_server).embed(["hello"], model="test-model")
        context = span.get_span_context()
    expected = f"00-{context.trace_id:032x}-{context.span_id:016x}-{int(context.trace_flags):02x}"
    assert embeddings_server.traceparents == [[expected]]


def test_explicit_traceparent_beats_active_span(embeddings_server):
    tracer = _sdk_tracer()
    with tracer.start_as_current_span("caller"):
        _client(embeddings_server).embed(
            ["hello"],
            model="test-model",
            preference=RequestProcessingPreference(traceparent=PARENT),
        )
    assert embeddings_server.traceparents == [[PARENT]]


def test_tracestate_preference_is_forwarded(embeddings_server):
    _client(embeddings_server).embed(
        ["hello"],
        model="test-model",
        preference=RequestProcessingPreference(
            traceparent=PARENT, tracestate="vendor=opaque"
        ),
    )
    assert embeddings_server.traceparents == [[PARENT]]
    assert embeddings_server.tracestates == [["vendor=opaque"]]


def test_tracestate_without_traceparent_raises_value_error(embeddings_server):
    with pytest.raises(ValueError, match="tracestate"):
        _client(embeddings_server).embed(
            ["hello"],
            model="test-model",
            preference=RequestProcessingPreference(tracestate="vendor=opaque"),
        )
    assert embeddings_server.traceparents == []


def test_active_span_tracestate_and_unsampled_flag_are_forwarded(embeddings_server):
    trace = pytest.importorskip("opentelemetry.trace")
    span_context = trace.SpanContext(
        trace_id=0x4BF92F3577B34DA6A3CE929D0E0E4736,
        span_id=0x00F067AA0BA902B7,
        is_remote=True,
        trace_flags=trace.TraceFlags(0),
        trace_state=trace.TraceState([("vendor", "opaque")]),
    )
    with trace.use_span(trace.NonRecordingSpan(span_context)):
        _client(embeddings_server).embed(["hello"], model="test-model")
    assert embeddings_server.traceparents == [
        ["00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-00"]
    ]
    assert embeddings_server.tracestates == [["vendor=opaque"]]


def test_extra_headers_traceparent_beats_active_span(embeddings_server):
    tracer = _sdk_tracer()
    with tracer.start_as_current_span("caller"):
        _client(embeddings_server).embed(
            ["hello"],
            model="test-model",
            preference=RequestProcessingPreference(
                extra_headers={"Traceparent": PARENT}
            ),
        )
    assert embeddings_server.traceparents == [[PARENT]]


def test_conflicting_traceparents_raise_value_error(embeddings_server):
    other = "00-99999999999999999999999999999999-aaaaaaaaaaaaaaaa-01"
    with pytest.raises(ValueError, match="set it in one place"):
        _client(embeddings_server).embed(
            ["hello"],
            model="test-model",
            preference=RequestProcessingPreference(
                traceparent=PARENT, extra_headers={"traceparent": other}
            ),
        )
    assert embeddings_server.traceparents == []
