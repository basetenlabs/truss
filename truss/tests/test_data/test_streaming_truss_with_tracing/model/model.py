import json
import pathlib
import time
from typing import Any, Generator, Sequence

import opentelemetry.sdk.resources as resources
import opentelemetry.sdk.trace as sdk_trace
import opentelemetry.sdk.trace.export as trace_export
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor


class JSONFileExporter(trace_export.SpanExporter):
    # Copied form truss/templates/server/common/tracing.py.

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


trace.set_tracer_provider(
    TracerProvider(resource=Resource.create({resources.SERVICE_NAME: "UserModel"}))
)
tracer = trace.get_tracer(__name__)
file_exporter = JSONFileExporter(pathlib.Path("/tmp/otel_user_traces.ndjson"))
span_processor = SimpleSpanProcessor(file_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)


class Model:
    def __init__(self, **kwargs) -> None:
        self._data_dir = kwargs["data_dir"]
        self._config = kwargs["config"]
        self._secrets = kwargs["secrets"]
        self._model = None

    @tracer.start_as_current_span("load_model")
    def load(self):
        pass

    @tracer.start_as_current_span("predict")
    def predict(self, model_input: Any) -> Generator[str, None, None]:
        with tracer.start_as_current_span("start-predict") as span:

            def inner():
                time.sleep(0.02)
                for i in range(5):
                    span.add_event("yield")
                    yield str(i)

            return inner()
