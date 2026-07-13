import json
import logging
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import opentelemetry.sdk.trace as sdk_trace
import pytest


@pytest.fixture
def load_telemetry():
    server_dir = Path(__file__).resolve().parents[3] / "templates" / "server"
    sys.path.insert(0, str(server_dir))
    try:
        import load_telemetry

        yield load_telemetry
    finally:
        sys.path.remove(str(server_dir))


@pytest.fixture
def telemetry_and_log(load_telemetry, caplog):
    caplog.set_level(logging.INFO, logger="test_load_telemetry")
    logger = logging.getLogger("test_load_telemetry")
    tracer = sdk_trace.TracerProvider().get_tracer(__name__)
    telemetry = load_telemetry.LoadTelemetry(logger, tracer)

    def emitted_fields() -> dict:
        lines = [
            r.message for r in caplog.records if "model load telemetry:" in r.message
        ]
        assert len(lines) == 1, f"expected one telemetry line, got {lines}"
        return json.loads(lines[0].split("model load telemetry:", 1)[1])

    return telemetry, emitted_fields


def test_stages_and_unattributed_time(telemetry_and_log):
    telemetry, emitted_fields = telemetry_and_log

    with telemetry.stage("setup"):
        time.sleep(0.02)
    with telemetry.stage("model_load"):
        time.sleep(0.05)
    telemetry.finalize(total_ms=100.0)

    fields = emitted_fields()
    assert fields["setup"] >= 20
    assert fields["model_load"] >= 50
    assert fields["load_total_ms"] == 100.0
    # unattributed = total - sum(stages); never negative
    assert 0 <= fields["unattributed_ms"] <= 100 - 70
    assert "torch_compile_ms" not in fields, "no torch imported -> no compile field"
    assert "gpu_mem_allocated_gb" not in fields


def test_stage_exception_still_records_duration_and_propagates(telemetry_and_log):
    telemetry, _ = telemetry_and_log
    with pytest.raises(RuntimeError):
        with telemetry.stage("model_load"):
            raise RuntimeError("customer load failed")
    assert "model_load" in telemetry._stage_ms


def test_torch_compile_delta_attributed_to_load(
    telemetry_and_log, load_telemetry, monkeypatch
):
    telemetry, emitted_fields = telemetry_and_log
    dynamo_utils = SimpleNamespace(
        compilation_time_metrics={"entire_frame_compile": [1.0]}
    )
    monkeypatch.setitem(sys.modules, "torch._dynamo.utils", dynamo_utils)

    # 1.0s of compile happened before load (e.g. during imports); snapshot it.
    telemetry.snapshot_before_model_load()
    # load() triggers 2.5s more compilation.
    dynamo_utils.compilation_time_metrics["entire_frame_compile"].append(2.5)
    telemetry.finalize(total_ms=5000.0)

    assert emitted_fields()["torch_compile_ms"] == 2500.0


def test_torch_compile_fallback_key_ladder(load_telemetry, monkeypatch):
    # Unknown frame-level keys: fall back to max-of-sums, never a double
    # counting grand total.
    dynamo_utils = SimpleNamespace(
        compilation_time_metrics={"phase_a": [1.0, 2.0], "phase_b": [0.5]}
    )
    monkeypatch.setitem(sys.modules, "torch._dynamo.utils", dynamo_utils)
    assert load_telemetry._torch_compile_seconds() == 3.0


def test_gpu_memory_only_read_when_cuda_initialized(load_telemetry, monkeypatch):
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_initialized=lambda: False,
            memory_allocated=lambda i: pytest.fail(
                "memory_allocated must not be called when CUDA is uninitialized "
                "(it would trigger CUDA init)"
            ),
            device_count=lambda: 1,
        )
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    assert load_telemetry._gpu_memory_allocated_bytes() is None

    fake_torch.cuda = SimpleNamespace(
        is_initialized=lambda: True,
        memory_allocated=lambda i: 2 * 1024**3,
        device_count=lambda: 2,
    )
    assert load_telemetry._gpu_memory_allocated_bytes() == 4 * 1024**3


def test_gpu_memory_delta_attributed_to_load(
    telemetry_and_log, load_telemetry, monkeypatch
):
    telemetry, emitted_fields = telemetry_and_log
    allocated = {"value": 1 * 1024**3}
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_initialized=lambda: True,
            memory_allocated=lambda i: allocated["value"],
            device_count=lambda: 1,
        )
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    telemetry.snapshot_before_model_load()
    allocated["value"] = 72 * 1024**3
    telemetry.finalize(total_ms=1000.0)

    fields = emitted_fields()
    assert fields["gpu_mem_allocated_gb"] == 72.0
    assert fields["gpu_mem_delta_gb"] == 71.0


def test_finalize_never_raises(load_telemetry, monkeypatch, caplog):
    caplog.set_level(logging.WARNING, logger="test_load_telemetry")
    logger = logging.getLogger("test_load_telemetry")
    tracer = sdk_trace.TracerProvider().get_tracer(__name__)
    telemetry = load_telemetry.LoadTelemetry(logger, tracer)
    monkeypatch.setattr(
        load_telemetry,
        "_torch_compile_seconds",
        lambda: (_ for _ in ()).throw(RuntimeError("introspection broke")),
    )

    telemetry.finalize(total_ms=100.0)  # must not raise

    assert any("finalize failed" in r.message for r in caplog.records)
