import signal
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple, cast

from rich.columns import Columns
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.text import Text

from truss.cli.train.poller import TrainingPollerMixin
from truss.remote.baseten.api import BasetenApi

METRICS_POLL_INTERVAL_SEC = 30


class MetricsWatcher(TrainingPollerMixin):
    live: Optional[Live]

    def __init__(self, api: BasetenApi, project_id: str, job_id: str, console: Console):
        super().__init__(api, project_id, job_id, console)

        self.live = None
        signal.signal(signal.SIGINT, self._handle_sigint)

    def _handle_sigint(self, signum: int, frame: Any) -> None:
        if self.live:
            self.live.stop()
        msg = f"\n\nExiting training job metrics. To stop the job, run `truss train stop --job-id {self.job_id}`"
        self.console.print(msg, style="yellow")
        raise KeyboardInterrupt()

    def _format_bytes(self, bytes_val: float) -> Tuple[str, str]:
        """Convert bytes to human readable format"""
        color_map = {"MB": "green", "GB": "cyan", "TB": "magenta"}
        unit = "MB"
        if bytes_val > 1024 * 1024 * 1024 * 1024:
            unit = "TB"
        elif bytes_val > 1024 * 1024 * 1024:
            unit = "GB"

        if unit == "MB":
            return f"{bytes_val / (1024 * 1024):.2f} MB", color_map[unit]
        elif unit == "GB":
            return f"{bytes_val / (1024 * 1024 * 1024):.2f} GB", color_map[unit]
        return f"{bytes_val:.2f} bytes", color_map[unit]

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

    def create_metrics_table(self, metrics_data: Dict) -> Columns:
        """Create a Rich table with the metrics"""
        compute_table = self._create_compute_table(metrics_data)
        storage_table = self._maybe_create_storage_table(metrics_data)
        tables = [compute_table]
        if storage_table:
            tables.append(storage_table)
        return Columns(tables, title="Training Job Metrics")

    def _create_compute_table(self, metrics_data: Dict) -> Table:
        table = Table(title="Compute Metrics")
        table.add_column("Metric")
        table.add_column("Value")

        # Add timestamp if available
        cpu_usage_data = metrics_data.get("cpu_usage", [])
        if cpu_usage_data and len(cpu_usage_data) > 0:
            latest_timestamp = cpu_usage_data[-1].get("timestamp")
            if latest_timestamp:
                table.add_row("Timestamp", latest_timestamp)
                table.add_section()

        # CPU metrics
        cpu_usage = self._get_latest_metric(metrics_data.get("cpu_usage", []))
        if cpu_usage is not None:
            table.add_row("CPU Usage", f"{cpu_usage:.2f} cores")

        cpu_memory = self._get_latest_metric(
            metrics_data.get("cpu_memory_usage_bytes", [])
        )
        if cpu_memory is not None:
            formatted_value, color = self._format_bytes(cpu_memory)
            table.add_row("CPU Memory", Text(formatted_value, style=color))

        # Add separator after CPU metrics
        table.add_section()

        # GPU metrics - grouped by GPU ID
        gpu_metrics = metrics_data.get("gpu_utilization", {})
        gpu_memory = metrics_data.get("gpu_memory_usage_bytes", {})

        for gpu_id in sorted(set(gpu_metrics.keys()) | set(gpu_memory.keys())):
            # Add GPU utilization
            latest_util = self._get_latest_metric(gpu_metrics.get(gpu_id, []))
            if latest_util is not None:
                table.add_row(f"GPU {gpu_id} Usage", f"{latest_util * 100:.1f}%")

            # Add GPU memory right after its utilization
            latest_memory = self._get_latest_metric(gpu_memory.get(gpu_id, []))
            if latest_memory is not None:
                formatted_value, color = self._format_bytes(latest_memory)
                table.add_row(
                    f"GPU {gpu_id} Memory", Text(formatted_value, style=color)
                )

            # Add separator after each GPU's metrics (except for the last one)
            if gpu_id != max(set(gpu_metrics.keys()) | set(gpu_memory.keys())):
                table.add_section()

        # Add separator before storage metrics
        if gpu_metrics or gpu_memory:
            table.add_section()
        return table

    def _maybe_create_storage_table(self, metrics_data: Dict) -> Optional[Table]:
        ephemeral_storage_metrics = metrics_data.get("ephemeral_storage")
        cache_storage_metrics = metrics_data.get("cache")
        if ephemeral_storage_metrics or cache_storage_metrics:
            storage_table = Table(title="Storage Metrics")
            storage_table.add_column("Storage Type")
            storage_table.add_column("Usage")
            storage_table.add_column("Utilization")
            did_add_ephemeral = self._maybe_format_storage_table_row(
                storage_table, "Ephemeral Storage", ephemeral_storage_metrics
            )
            did_add_cache = self._maybe_format_storage_table_row(
                storage_table, "Cache Storage", cache_storage_metrics
            )
            if did_add_ephemeral or did_add_cache:
                return storage_table
        return None

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
                    self.console.print(
                        f"Error fetching metrics: {e}: {traceback.format_exc()}",
                        style="red",
                    )
                    break
        self.after_polling()
