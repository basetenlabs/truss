import contextlib
import functools
import json
import logging
import os
import pathlib
import time
from typing import Iterator, List, Sequence

import opentelemetry.exporter.otlp.proto.grpc.trace_exporter as oltp_exporter
import opentelemetry.sdk.resources as resources
import opentelemetry.sdk.trace as sdk_trace
import opentelemetry.sdk.trace.export as trace_export
from opentelemetry import context, trace

logger = logging.getLogger(__name__)


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


@functools.lru_cache(maxsize=1)
def get_truss_tracer() -> trace.Tracer:
    """Creates a cached tracer (i.e. runtime-singleton) to be used for truss
    internal tracing.

    The goal is to separate truss-internal tracing instrumentation
    completely from potential user-defined tracing - see also `detach_context` below.

    """
    span_processors: List[sdk_trace.SpanProcessor] = []
    if otlp_endpoint := os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
        otlp_exporter = oltp_exporter.OTLPSpanExporter(endpoint=otlp_endpoint)
        otlp_processor = sdk_trace.export.BatchSpanProcessor(otlp_exporter)
        span_processors.append(otlp_processor)

    if tracing_log_file := os.getenv("OTEL_TRACING_NDJSON_FILE"):
        json_file_exporter = JSONFileExporter(pathlib.Path(tracing_log_file))
        file_processor = sdk_trace.export.SimpleSpanProcessor(json_file_exporter)
        span_processors.append(file_processor)

    if span_processors:
        logger.info("Instantiating truss tracer.")
        resource = resources.Resource.create({resources.SERVICE_NAME: "TrussServer"})
        trace_provider = sdk_trace.TracerProvider(resource=resource)
        for sp in span_processors:
            trace_provider.add_span_processor(sp)
        tracer = trace_provider.get_tracer("truss_server")
    else:
        logger.info("Using no-op tracing.")
        tracer = sdk_trace.NoOpTracer()

    return tracer


@contextlib.contextmanager
def detach_context() -> Iterator[None]:
    """Breaks opentelemetry's context propagation.

    The goal is to separate truss-internal tracing instrumentation
    completely from potential user-defined tracing. Opentelemetry has a global state
    that makes "outer" span-contexts parents of nested spans. If user-code in a
    truss model also uses tracing, these traces could easily become polluted with our
    internal contexts. Therefore, all user code (predict and pre/post-processing) should
    be wrapped in this context for isolation.
    """
    current_context = context.get_current()
    # Set the current context to an invalid span context, effectively clearing it.
    # This makes sure inside the context a new root is context is created.
    transient_token = context.attach(
        trace.set_span_in_context(
            trace.INVALID_SPAN,
            trace.INVALID_SPAN_CONTEXT,  # type: ignore[arg-type]
        )
    )
    try:
        yield
    finally:
        # Reattach original context.
        context.detach(transient_token)
        context.attach(current_context)


@contextlib.contextmanager
def section_as_event(span: sdk_trace.Span, section_name: str) -> Iterator[None]:
    """Helper to record the start and end of a sections as events and the duration.

    Note that events are much cheaper to create than dedicated spans.
    """
    t0 = time.time()
    span.add_event(f"start-{section_name}")
    try:
        yield
    finally:
        t1 = time.time()
        span.add_event(f"done-{section_name}", attributes={"duration_sec": t1 - t0})
