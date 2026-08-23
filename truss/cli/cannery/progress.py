from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional


def event_kind(event: Mapping[str, Any]) -> Optional[str]:
    for key in ("type", "event", "kind"):
        value = event.get(key)
        if isinstance(value, str):
            return value
    return None


def _event_values(event: Mapping[str, Any]) -> Mapping[str, Any]:
    progress = event.get("progress")
    if isinstance(progress, dict):
        return {**event, **progress}
    return event


def _format_progress_event(event: Mapping[str, Any]) -> Optional[str]:
    values = _event_values(event)
    operation = values.get("operation") or values.get("command") or "transfer"
    phase = values.get("phase")
    label = f"Cannery {operation}"
    if isinstance(phase, str):
        label += f" ({phase})"

    counters = []
    for noun in ("files", "chunks", "bytes"):
        done = values.get(f"{noun}_done")
        if done is None:
            done = values.get(f"completed_{noun}")
        total = values.get(f"{noun}_total")
        if total is None:
            total = values.get(f"total_{noun}")
        if done is not None and total is not None:
            counters.append(f"{done}/{total} {noun}")
        elif done is not None:
            counters.append(f"{done} {noun}")

    if not counters:
        completed = values.get("completed")
        total = values.get("total")
        unit = values.get("unit") or "items"
        if completed is not None and total is not None:
            counters.append(f"{completed}/{total} {unit}")
        elif completed is not None:
            counters.append(f"{completed} {unit}")

    if not counters:
        return None
    return f"{label}: {', '.join(counters)}"


def _format_status_event(event: Mapping[str, Any]) -> Optional[str]:
    operation = event.get("operation") or event.get("command")
    phase = event.get("phase")
    status = event.get("status") or event.get("state")
    typed_values = [
        value for value in (operation, phase, status) if isinstance(value, str)
    ]
    if not typed_values:
        return None
    return f"Cannery {' — '.join(typed_values)}"


def format_event(event: Mapping[str, Any]) -> Optional[str]:
    kind = event_kind(event)
    if kind == "progress":
        return _format_progress_event(event)
    if kind in {"started", "status", "warning"}:
        return _format_status_event(event)
    return None


@dataclass
class BoundedProgressState:
    """Only state needed to explain the latest protocol outcome."""

    last_phase: Optional[str] = None
    terminal_error: Optional[Mapping[str, Any]] = None

    def observe(self, event: Mapping[str, Any]) -> None:
        values = _event_values(event)
        phase = values.get("phase")
        if isinstance(phase, str):
            self.last_phase = phase

        if event_kind(event) == "error":
            error = event.get("error")
            if isinstance(error, dict):
                self.terminal_error = {**event, **error}
            else:
                self.terminal_error = event
