"""Per-stage timing and torch introspection for model load.

Breaks the opaque model.load() time into named stages (weights, imports,
init, load) plus torch compile / GPU-memory attribution, emitted as one
structured log line and OTel spans. See the "Profile container_ready
portion of GPU Cold Start" project (RUN-544).

Two invariants:

- Telemetry is write-only: any internal failure degrades to missing
  fields and a warning, never to a failed or slower load.
- torch is never imported here. It is looked up in sys.modules, so it is
  only observed when the customer's own code already imported it; models
  without torch pay nothing, and the server does not spend seconds
  importing torch on their behalf.
"""

import contextlib
import json
import logging
import sys
import time
from typing import Dict, Iterator, Optional

import opentelemetry.sdk.trace as sdk_trace


class LoadTelemetry:
    """Times the stages of ModelWrapper._load_impl and snapshots torch
    compile/CUDA state around the customer's load()."""

    def __init__(self, logger: logging.Logger, tracer: sdk_trace.Tracer) -> None:
        self._logger = logger
        self._tracer = tracer
        self._stage_ms: Dict[str, float] = {}
        self._pre_load_compile_s: Optional[float] = None
        self._pre_load_gpu_mem_bytes: Optional[int] = None

    @contextlib.contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Times a stage of the load path; also emits it as a span so the
        breakdown reaches the trace pipeline when tracing is enabled."""
        start = time.perf_counter()
        with self._tracer.start_as_current_span(f"load-{name}"):
            try:
                yield
            finally:
                self._stage_ms[name] = _round_ms(time.perf_counter() - start)

    def snapshot_before_model_load(self) -> None:
        """Captures torch compile/GPU baselines right before the customer's
        load() so finalize() can attribute deltas to it. The customer's
        module imports have already run by this point, so torch (if used)
        is importable state we can observe."""
        try:
            self._pre_load_compile_s = _torch_compile_seconds()
            self._pre_load_gpu_mem_bytes = _gpu_memory_allocated_bytes()
        except Exception:
            self._logger.warning(
                "load telemetry: pre-load snapshot failed", exc_info=True
            )

    def finalize(self, total_ms: float) -> None:
        """Emits the load breakdown as a single structured log line."""
        try:
            fields: Dict[str, object] = dict(self._stage_ms)
            fields["load_total_ms"] = round(total_ms, 1)

            compile_s = _torch_compile_seconds()
            if compile_s is not None:
                fields["torch_compile_ms"] = _round_ms(
                    compile_s - (self._pre_load_compile_s or 0.0)
                )

            gpu_mem_bytes = _gpu_memory_allocated_bytes()
            if gpu_mem_bytes is not None:
                fields["gpu_mem_allocated_gb"] = round(gpu_mem_bytes / 1024**3, 2)
                if self._pre_load_gpu_mem_bytes is not None:
                    fields["gpu_mem_delta_gb"] = round(
                        (gpu_mem_bytes - self._pre_load_gpu_mem_bytes) / 1024**3, 2
                    )

            attributed_ms = sum(self._stage_ms.values())
            fields["unattributed_ms"] = round(max(total_ms - attributed_ms, 0.0), 1)

            self._logger.info(
                f"model load telemetry: {json.dumps(fields, sort_keys=True)}"
            )
        except Exception:
            self._logger.warning("load telemetry: finalize failed", exc_info=True)


def _round_ms(seconds: float) -> float:
    return round(seconds * 1000, 1)


def _torch_compile_seconds() -> Optional[float]:
    """Total torch.compile backend wall time observed so far, or None when
    torch dynamo is absent.

    compilation_time_metrics is a semi-private dict of phase name -> list
    of durations whose keys have moved across torch versions, so everything
    is feature-detected. Frame-level phases bound total compile wall time;
    nested phases would double count, hence the preferred-key ladder with a
    max-of-sums fallback rather than a grand total.
    """
    dynamo_utils = sys.modules.get("torch._dynamo.utils")
    if dynamo_utils is None:
        return None
    metrics = getattr(dynamo_utils, "compilation_time_metrics", None)
    if not isinstance(metrics, dict):
        return None
    if not metrics:
        return 0.0
    for frame_level_key in ("entire_frame_compile", "_compile.compile_inner"):
        durations = metrics.get(frame_level_key)
        if durations:
            return float(sum(durations))
    return float(max((sum(v) for v in metrics.values() if v), default=0.0))


def _gpu_memory_allocated_bytes() -> Optional[int]:
    """Total CUDA memory allocated by torch across devices, or None when
    unobservable. Only reads state when CUDA is already initialized —
    memory_allocated() on an uninitialized runtime would trigger CUDA init,
    and telemetry must never alter load behavior."""
    torch = sys.modules.get("torch")
    if torch is None:
        return None
    try:
        cuda = torch.cuda
        if not cuda.is_initialized():
            return None
        return int(sum(cuda.memory_allocated(i) for i in range(cuda.device_count())))
    except Exception:
        return None
