import contextlib
import json
import logging
import os
import pathlib
import time
from typing import Iterator, List, Optional, Sequence

import opentelemetry.exporter.otlp.proto.http.trace_exporter as oltp_exporter
import opentelemetry.sdk.resources as resources
import opentelemetry.sdk.trace as sdk_trace
import opentelemetry.sdk.trace.export as trace_export
from opentelemetry import context, trace
from shared import secrets_resolver

logger = logging.getLogger(__name__)

ATTR_NAME_DURATION = "duration_sec"
OTEL_EXPORTER_OTLP_ENDPOINT = "OTEL_EXPORTER_OTLP_ENDPOINT"
# Writing trace data to a file is only intended for testing / debugging.
OTEL_TRACING_NDJSON_FILE = "OTEL_TRACING_NDJSON_FILE"
# Exporting trace data to a public honeycomb instance (not our cluster collector)
# is intended only for testing / debugging.
HONEYCOMB_DATASET = "HONEYCOMB_DATASET"
HONEYCOMB_API_KEY = "HONEYCOMB_API_KEY"

DEFAULT_ENABLE_TRACING_DATA = False  # This should be in sync with truss_config.py.


class JSONFileExporter(trace_export.SpanExporter):
    """Writes spans to newline-delimited JSON file for debugging / testing."""

    def __init__(self, file_path: pathlib.Path):
        self._file = file_path.open("a")

    def export(
        self, spans: Sequence[sdk_trace.ReadableSpan]
    ) -> trace_export.SpanExportResult:
        for span in spans:
            # Get rid of newlines and whitespace.
            self._file.write(json.dumps(json.loads(span.to_json())))
            self._file.write("\n")
        self._file.flush()
        return trace_export.SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        self._file.close()


_truss_tracer: Optional[trace.Tracer] = None


def get_truss_tracer(secrets: secrets_resolver.Secrets, config) -> trace.Tracer:
    """Creates a cached tracer (i.e. runtime-singleton) to be used for truss
    internal tracing.

    The goal is to separate truss-internal tracing instrumentation
    completely from potential user-defined tracing - see also `detach_context` below.

    """
    enable_tracing_data = config.get("runtime", {}).get(
        "enable_tracing_data", DEFAULT_ENABLE_TRACING_DATA
    )

    global _truss_tracer
    if _truss_tracer:
        return _truss_tracer

    span_processors: List[sdk_trace.SpanProcessor] = []
    if otlp_endpoint := os.getenv(OTEL_EXPORTER_OTLP_ENDPOINT):
        if enable_tracing_data:
            logger.info(f"Exporting trace data to {OTEL_EXPORTER_OTLP_ENDPOINT}.")
        otlp_exporter = oltp_exporter.OTLPSpanExporter(endpoint=otlp_endpoint)
        otlp_processor = sdk_trace.export.BatchSpanProcessor(otlp_exporter)
        span_processors.append(otlp_processor)

    if tracing_log_file := os.getenv(OTEL_TRACING_NDJSON_FILE):
        if enable_tracing_data:
            logger.info(f"Exporting trace data to file `{tracing_log_file}`.")
        json_file_exporter = JSONFileExporter(pathlib.Path(tracing_log_file))
        file_processor = sdk_trace.export.SimpleSpanProcessor(json_file_exporter)
        span_processors.append(file_processor)

    if (
        honeycomb_dataset := os.getenv(HONEYCOMB_DATASET)
    ) and HONEYCOMB_API_KEY in secrets:
        honeycomb_api_key = secrets[HONEYCOMB_API_KEY]
        logger.info("Exporting trace data to honeycomb.")
        honeycomb_exporter = oltp_exporter.OTLPSpanExporter(
            endpoint="https://api.honeycomb.io/v1/traces",
            headers={
                "x-honeycomb-team": honeycomb_api_key,
                "x-honeycomb-dataset": honeycomb_dataset,
            },
        )
        honeycomb_processor = sdk_trace.export.BatchSpanProcessor(honeycomb_exporter)
        span_processors.append(honeycomb_processor)

    if span_processors and enable_tracing_data:
        logger.info("Instantiating truss tracer.")
        resource = resources.Resource.create({resources.SERVICE_NAME: "truss-server"})
        trace_provider = sdk_trace.TracerProvider(resource=resource)
        for sp in span_processors:
            trace_provider.add_span_processor(sp)
        tracer = trace_provider.get_tracer("truss_server")
    else:
        if enable_tracing_data:
            logger.info(
                "Using no-op tracing (tracing is enabled, but no exporters configured)."
            )
        else:
            logger.info("Using no-op tracing (tracing was disabled).")

        tracer = sdk_trace.NoOpTracer()

    _truss_tracer = tracer
    return _truss_tracer


@contextlib.contextmanager
def section_as_event(
    span: sdk_trace.Span, section_name: str, detach: bool = False
) -> Iterator[Optional[trace.Context]]:
    """Helper to record the start and end of a sections as events and the duration.

    Note that events are much cheaper to create than dedicated spans.

    Optionally detaches the OpenTelemetry context to isolate tracing.
    This intentionally breaks opentelemetry's context propagation.

    The goal is to separate truss-internal tracing instrumentation
    completely from potential user-defined tracing. Opentelemetry has a global state
    that makes "outer" span-contexts parents of nested spans. If user-code in a
    truss model also uses tracing, these traces could easily become polluted with our
    internal contexts. Therefore, all user code (predict and pre/post-processing) should
    be wrapped in this context for isolation.
    """
    t0 = time.time()
    span.add_event(f"start: {section_name}")

    maybe_ctx: Optional[tuple[context.Context, context.Token]] = None

    if detach:
        maybe_ctx = (
            context.get_current(),
            context.attach(trace.set_span_in_context(trace.INVALID_SPAN)),
        )

    try:
        yield maybe_ctx[0] if maybe_ctx else None
    finally:
        t1 = time.time()
        span.add_event(
            f"done: {section_name}", attributes={ATTR_NAME_DURATION: t1 - t0}
        )
        if maybe_ctx:
            context.detach(maybe_ctx[1])
            context.attach(maybe_ctx[0])
