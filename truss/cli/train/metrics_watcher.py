import signal
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

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

    def _get_latest_metric(self, metrics: List[Dict]) -> Optional[float]:
        """Get the most recent metric value"""
        if not metrics:
            return None
        return metrics[-1].get("value")

    def create_metrics_table(self, metrics_data: Dict) -> Table:
        """Create a Rich table with the metrics"""
        table = Table(title="Training Job Metrics")
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
