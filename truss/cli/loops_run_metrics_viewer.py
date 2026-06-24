"""Display layer for ``truss loops runs metrics``.

Owns the cli-table layout, NDJSON emission, and live-tail loop. Kept
separate from ``loops_commands.py`` so the command body reads as wiring +
dispatch.
"""

import json
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Callable, Dict, Iterator, List, Optional, TextIO

import rich.live
import rich.table

from truss.cli.utils.output import console

DEFAULT_METRICS_REFRESH_SECONDS = 30

_SPARKLINE_BLOCKS = "▁▂▃▄▅▆▇█"
_SPARKLINE_WIDTH = 24


def render_metrics_snapshot(snapshot: Dict[str, Any]) -> None:
    """Render a single metrics snapshot as a Rich table to the console."""
    console.print(_build_metrics_table(snapshot))


def tail_metrics_table(
    fetch_snapshot: Callable[[], Dict[str, Any]],
    *,
    refresh_rate_seconds: int,
    run_id: str,
) -> None:
    """Stream cli-table updates until the caller interrupts with Ctrl+C."""
    with rich.live.Live(auto_refresh=False, console=console, screen=False) as live:
        try:
            while True:
                live.update(_build_metrics_table(fetch_snapshot()), refresh=True)
                time.sleep(refresh_rate_seconds)
        except KeyboardInterrupt:
            live.stop()
            console.print(
                f"\nStopped watching metrics for run {run_id}.", style="yellow"
            )


def emit_json_snapshots(
    fetch_snapshot: Callable[[], Dict[str, Any]],
    *,
    tail: bool,
    refresh_rate_seconds: int,
    output_file: Optional[str],
    run_id: str,
) -> None:
    """Emit one JSON doc (snapshot) or NDJSON (tail) to stdout or ``output_file``."""
    with _open_metrics_sink(output_file) as sink:
        if not tail:
            sink.write(json.dumps(fetch_snapshot()) + "\n")
            sink.flush()
            return
        try:
            while True:
                sink.write(json.dumps(fetch_snapshot()) + "\n")
                sink.flush()
                time.sleep(refresh_rate_seconds)
        except KeyboardInterrupt:
            print(f"Stopped streaming metrics for run {run_id}.", file=sys.stderr)


def _build_metrics_table(snapshot: Dict[str, Any]) -> rich.table.Table:
    window = snapshot["window"]
    table = rich.table.Table(
        show_header=True,
        header_style="bold magenta",
        title=(
            f"Metrics for Loops run [cyan]{snapshot['run_id']}[/cyan]  "
            f"({_short_iso(window['start'])} → {_short_iso(window['end'])})"
        ),
        box=rich.table.box.ROUNDED,
        border_style="blue",
    )
    table.add_column("Component", style="green")
    table.add_column("Deployment", style="cyan")
    table.add_column("Request volume (req/s)", justify="right")
    table.add_column("Concurrent requests", justify="right")
    table.add_column("Trend", justify="left")

    trainer = snapshot["trainer"]
    table.add_row(
        "Trainer",
        snapshot["trainer_deployment_id"],
        _fmt_latest(_latest_value(trainer["request_volume"])),
        _fmt_latest(_latest_value(trainer["concurrent_requests"])),
        _sparkline(trainer["request_volume"]),
    )

    sampler_id = snapshot["sampler_deployment_id"]
    sampler = snapshot["sampler"]
    if sampler_id:
        table.add_row(
            "Sampler",
            sampler_id,
            _fmt_latest(_latest_value(sampler["request_volume"])),
            _fmt_latest(_latest_value(sampler["concurrent_requests"])),
            _sparkline(sampler["request_volume"]),
        )
    else:
        table.add_row("Sampler", "—", "—", "—", "")
    return table


def _latest_value(points: List[Dict[str, Any]]) -> Optional[float]:
    if not points:
        return None
    return points[-1].get("value")


def _sparkline(points: List[Dict[str, Any]]) -> str:
    values = [p["value"] for p in points if p.get("value") is not None]
    if not values:
        return ""
    width = min(_SPARKLINE_WIDTH, len(values))
    if len(values) > width:
        bucket = len(values) / width
        bucketed = []
        for i in range(width):
            lo_idx = int(i * bucket)
            hi_idx = max(int((i + 1) * bucket), lo_idx + 1)
            chunk = values[lo_idx:hi_idx]
            bucketed.append(sum(chunk) / len(chunk))
    else:
        bucketed = values
    lo, hi = min(bucketed), max(bucketed)
    if hi - lo <= 0:
        return _SPARKLINE_BLOCKS[len(_SPARKLINE_BLOCKS) // 2] * len(bucketed)
    return "".join(
        _SPARKLINE_BLOCKS[int((v - lo) / (hi - lo) * (len(_SPARKLINE_BLOCKS) - 1))]
        for v in bucketed
    )


def _short_iso(iso: str) -> str:
    try:
        return (
            datetime.fromisoformat(iso.replace("Z", "+00:00"))
            .astimezone()
            .strftime("%Y-%m-%d %H:%M:%S")
        )
    except ValueError:
        return iso


def _fmt_latest(value: Optional[float]) -> str:
    if value is None:
        return "—"
    return f"{value:.2f}"


@contextmanager
def _open_metrics_sink(output_file: Optional[str]) -> Iterator[TextIO]:
    if output_file is None or output_file == "-":
        yield sys.stdout
        return
    with open(output_file, "w") as f:
        yield f
