import signal
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple, cast

from rich.columns import Columns
from rich.layout import Layout
from rich.live import Live
from rich.table import Table
from rich.text import Text

from truss.cli.train.poller import TrainingPollerMixin
from truss.cli.utils import common
from truss.cli.utils.output import console
from truss.remote.baseten.api import BasetenApi

METRICS_POLL_INTERVAL_SEC = 30


class MetricsWatcher(TrainingPollerMixin):
    live: Optional[Live]

    def __init__(self, api: BasetenApi, project_id: str, job_id: str):
        super().__init__(api, project_id, job_id)

        self.live = None
        signal.signal(signal.SIGINT, self._handle_sigint)

    def _handle_sigint(self, signum: int, frame: Any) -> None:
        if self.live:
            self.live.stop()
        msg = f"\n\nExiting training job metrics. To stop the job, run `truss train stop --job-id {self.job_id}`"
        console.print(msg, style="yellow")
        raise KeyboardInterrupt()

    def _format_bytes(self, bytes_val: float) -> Tuple[str, str]:
        """Convert bytes to human readable format"""
        default_color = "green"
        color_map = {"MB": "green", "GB": "cyan", "TB": "magenta"}
        unit = "B"
        if bytes_val > 1000 * 1000 * 1000 * 1000:
            unit = "TB"
        elif bytes_val > 1000 * 1000 * 1000:
            unit = "GB"
        elif bytes_val > 1000 * 1000:
            unit = "MB"
        color = color_map.get(unit, default_color)
        return (common.format_bytes_to_human_readable(int(bytes_val)), color)

    def _format_storage_utilization(self, utilization: float) -> Tuple[str, str]:
        percent = round(utilization * 100, 4)
        if percent > 90:
            return f"{percent}%", "red"
        elif percent > 70:
            return f"{percent}%", "yellow"
        return f"{percent}%", "green"

    def _get_latest_metric(self, metrics: List[Dict]) -> Optional[float]:
        """Get the most recent metric value"""
        if not metrics:
            return None
        return metrics[-1].get("value")

    def _get_latest_storage_metrics(
        self, storage_data: Optional[Dict[str, List[Dict]]]
    ) -> Optional[Tuple[int, float]]:
        if not storage_data:
            return None
        usage_data = storage_data.get("usage_bytes")
        utilization_data = storage_data.get("utilization")
        if not usage_data or not utilization_data:
            return None
        usage_value = usage_data[-1].get("value", None)
        utilization_value = utilization_data[-1].get("value", None)
        if not usage_value or not utilization_value:
            return None
        return cast(int, usage_value), cast(float, utilization_value)

    def _maybe_format_storage_table_row(
        self, table: Table, label: str, storage_data: Optional[Dict[str, List[Dict]]]
    ) -> bool:
        if not storage_data:
            return False
        maybe_values = self._get_latest_storage_metrics(storage_data)
        if not maybe_values:
            return False
        raw_usage, raw_utilization = maybe_values
        usage_value, usage_color = self._format_bytes(raw_usage)
        utilization_value, utilization_color = self._format_storage_utilization(
            raw_utilization
        )
        table.add_row(
            label,
            Text(usage_value, style=usage_color),
            Text(utilization_value, style=utilization_color),
        )
        return True

    def create_metrics_table(self, metrics_data: Dict) -> Layout:
        """Create a Rich table with the metrics"""
        tables = []

        timestamp = self._get_timestamp_from_metrics(metrics_data)

        node_tables = self._create_unified_node_metrics_tables(metrics_data)
        tables.extend(node_tables)

        storage_tables = self._create_storage_tables(metrics_data)
        tables.extend(storage_tables)

        columns = Columns(tables, title="Training Job Metrics")

        layout = Layout()

        if timestamp:
            from rich.panel import Panel

            layout.split_column(
                Layout(
                    Panel(
                        f"🕐 Last Updated: {timestamp}\n💡 Press Ctrl+C to exit",
                        style="bold cyan",
                    ),
                    size=4,
                ),
                Layout(columns),
            )
        else:
            layout.split_column(Layout(columns))

        return layout

    def _get_timestamp_from_metrics(self, metrics_data: Dict) -> Optional[str]:
        """Extract timestamp from metrics data for display"""
        # Try to get timestamp from per_node_metrics first. Fall back to main metrics if not there.
        per_node_metrics = metrics_data.get("per_node_metrics", [])
        if per_node_metrics and len(per_node_metrics) > 0:
            first_node_metrics = per_node_metrics[0].get("metrics", {})
            cpu_usage_data = first_node_metrics.get("cpu_usage", [])
            if cpu_usage_data and len(cpu_usage_data) > 0:
                timestamp = cpu_usage_data[-1].get("timestamp")
                if timestamp:
                    return common.format_localized_time(timestamp)

        cpu_usage_data = metrics_data.get("cpu_usage", [])
        if cpu_usage_data and len(cpu_usage_data) > 0:
            timestamp = cpu_usage_data[-1].get("timestamp")
            if timestamp:
                return common.format_localized_time(timestamp)

        return None

    def _create_unified_node_metrics_tables(self, metrics_data: Dict) -> List[Table]:
        """Create tables for node metrics, handling both single and multi-node scenarios"""
        tables = []

        per_node_metrics = metrics_data.get("per_node_metrics", [])

        if not per_node_metrics:
            # Job is likely just starting up - it takes some type for the
            # the metrics to become available after the job starts running.
            from rich.text import Text

            waiting_table = Table(title="Training Job Status")
            waiting_table.add_column("Status")
            waiting_table.add_column("Message")

            waiting_table.add_row(
                "Status",
                Text("⏳ Waiting for metrics to become available...", style="yellow"),
            )
            waiting_table.add_row(
                "Note",
                Text(
                    "Metrics will appear once the training job starts running.",
                    style="dim",
                ),
            )

            tables.append(waiting_table)
            return tables

        for node_metrics in per_node_metrics:
            node_id = node_metrics.get("node_id", "Unknown")
            metrics = node_metrics.get("metrics", {})

            if not metrics:
                continue

            table = self._create_node_table(node_id, metrics)
            tables.append(table)

        return tables

    def _create_node_table(self, node_id: str, metrics: Dict) -> Table:
        """Create a table for a single node's metrics"""
        table = Table(title=f"Node: {node_id}")
        table.add_column("Metric")
        table.add_column("Value")

        cpu_usage = self._get_latest_metric(metrics.get("cpu_usage", []))
        if cpu_usage is not None:
            table.add_row("CPU usage", f"{cpu_usage:.2f} cores")

        cpu_memory = self._get_latest_metric(metrics.get("cpu_memory_usage_bytes", []))
        if cpu_memory is not None:
            formatted_value, color = self._format_bytes(cpu_memory)
            table.add_row("CPU memory", Text(formatted_value, style=color))

        if cpu_usage is not None or cpu_memory is not None:
            table.add_section()

        gpu_utilization = metrics.get("gpu_utilization", {})
        gpu_memory = metrics.get("gpu_memory_usage_bytes", {})

        # API should return same GPU IDs for utilization and memory
        keys = gpu_utilization.keys()
        for idx, gpu_id in enumerate(keys):
            latest_util = self._get_latest_metric(gpu_utilization.get(gpu_id, []))
            if latest_util is not None:
                table.add_row(f"GPU {gpu_id} utilization", f"{latest_util * 100:.1f}%")

            latest_memory = self._get_latest_metric(gpu_memory.get(gpu_id, []))
            if latest_memory is not None:
                formatted_value, color = self._format_bytes(latest_memory)
                table.add_row(
                    f"GPU {gpu_id} memory", Text(formatted_value, style=color)
                )

            if idx != len(keys) - 1:
                table.add_section()

        ephemeral_storage = metrics.get("ephemeral_storage")
        if ephemeral_storage:
            if gpu_utilization or gpu_memory:
                table.add_section()

            usage_bytes = self._get_latest_metric(
                ephemeral_storage.get("usage_bytes", [])
            )
            utilization = self._get_latest_metric(
                ephemeral_storage.get("utilization", [])
            )

            if usage_bytes is not None:
                formatted_value, color = self._format_bytes(usage_bytes)
                table.add_row("Eph. storage usage", Text(formatted_value, style=color))

            if utilization is not None:
                utilization_percent = utilization * 100
                if utilization_percent > 90:
                    color = "red"
                elif utilization_percent > 70:
                    color = "yellow"
                else:
                    color = "green"
                table.add_row(
                    "Eph. storage utilization",
                    Text(f"{utilization_percent:.1f}%", style=color),
                )

        return table

    def _create_storage_tables(self, metrics_data: Dict) -> List[Table]:
        """Create storage tables - only cache per job (ephemeral is now in node tables)"""
        tables = []

        # Create cache storage table (job-level, shown once)
        cache_storage = metrics_data.get("cache")
        if cache_storage:
            table = self._create_cache_storage_table(cache_storage)
            if table:
                tables.append(table)

        return tables

    def _create_cache_storage_table(self, cache_storage: Dict) -> Optional[Table]:
        """Create table for cache storage metrics (job-level)"""
        usage_bytes = self._get_latest_metric(cache_storage.get("usage_bytes", []))
        utilization = self._get_latest_metric(cache_storage.get("utilization", []))

        if usage_bytes is None and utilization is None:
            return None

        table = Table(title="Cache storage")
        table.add_column("Storage Type")
        table.add_column("Usage")
        table.add_column("Utilization")

        self._maybe_format_storage_table_row(table, "Cache storage", cache_storage)

        return table

    def watch(self, refresh_rate: int = METRICS_POLL_INTERVAL_SEC):
        """Display continuously updating metrics"""
        self.before_polling()
        with Live(auto_refresh=False) as live:
            self.live = live
            while True:
                # our first instance of fetching metrics passes no explicit time range. We do this so that we can fetch metrics
                # for inactive jobs, using the job's completion time to set the time range.
                # Subsequent queries will fetch only the most recent data to avoid unnecessary load on VM
                metrics = self.api.get_training_job_metrics(
                    self.project_id, self.job_id
                )
                try:
                    # range of one minute since we only want the last recording
                    table = self.create_metrics_table(metrics)
                    live.update(table, refresh=True)
                    if not self.should_poll_again():
                        live.stop()
                        break
                    time.sleep(refresh_rate)
                    end_epoch_millis = int(time.time() * 1000)
                    start_epoch_millis = end_epoch_millis - 60 * 1000
                    metrics = self.api.get_training_job_metrics(
                        self.project_id,
                        self.job_id,
                        end_epoch_millis=end_epoch_millis,
                        start_epoch_millis=start_epoch_millis,
                    )
                    self.post_poll()
                except Exception as e:
                    live.stop()
                    console.print(
                        f"Error fetching metrics: {e}: {traceback.format_exc()}",
                        style="red",
                    )
                    break
        self.after_polling()
