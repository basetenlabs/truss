import time
from typing import Dict, List, Optional

from rich.console import Console
from rich.live import Live
from rich.table import Table

from truss.cli.training_poller import TrainingPollerMixin
from truss.remote.baseten.api import BasetenApi


class MetricsWatcher(TrainingPollerMixin):
    def __init__(self, api: BasetenApi, project_id: str, job_id: str, console: Console):
        super().__init__(api, project_id, job_id, console)

    def _format_bytes(self, bytes_val: float, unit: str = "MB") -> str:
        """Convert bytes to human readable format"""
        if unit == "MB":
            return f"{bytes_val / (1024 * 1024):.2f} MB"
        elif unit == "GB":
            return f"{bytes_val / (1024 * 1024 * 1024):.2f} GB"
        return f"{bytes_val:.2f} bytes"

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

        # CPU metrics
        cpu_usage = self._get_latest_metric(metrics_data.get("cpu_usage", []))
        if cpu_usage is not None:
            table.add_row("CPU Usage", f"{cpu_usage:.2f} cores")

        cpu_memory = self._get_latest_metric(
            metrics_data.get("cpu_memory_usage_bytes", [])
        )
        if cpu_memory is not None:
            table.add_row("CPU Memory", self._format_bytes(cpu_memory))

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
                table.add_row(f"GPU {gpu_id} Memory", self._format_bytes(latest_memory))

            # Add separator after each GPU's metrics (except for the last one)
            if gpu_id != max(set(gpu_metrics.keys()) | set(gpu_memory.keys())):
                table.add_section()

        # Add separator before storage metrics
        if gpu_metrics or gpu_memory:
            table.add_section()

        # Storage metrics
        storage = metrics_data.get("storage_metrics")
        if storage:
            table.add_row(
                "Disk Free",
                f"{storage.get('ephemeral_storage_available_gib', 0):.2f} GB",
            )
            table.add_row(
                "Disk Used", f"{storage.get('ephemeral_storage_used_gib', 0):.2f} GB"
            )

        return table

    def display_live_metrics(self, refresh_rate: int = 3):
        """Display continuously updating metrics"""
        self.before_polling()
        with Live(auto_refresh=False) as live:
            while True:
                try:
                    metrics = self.api.get_training_job_metrics(
                        self.project_id, self.job_id
                    )
                    table = self.create_metrics_table(metrics)
                    live.update(table, refresh=True)
                    if not self.should_poll_again():
                        live.stop()
                        break
                    time.sleep(refresh_rate)
                    self.post_poll()
                except KeyboardInterrupt:
                    live.stop()
                    self.console.print(
                        f"Exiting metrics display. To stop the job, run `truss train stop --job-id {self.job_id}`",
                        style="yellow",
                    )
                    break
                except Exception as e:
                    live.stop()
                    self.console.print(f"Error fetching metrics: {e}", style="red")
                    break
        self.after_polling()
