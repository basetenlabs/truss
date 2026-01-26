import atexit
import json
import logging
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from threading import Lock, Thread
from typing import Optional, Union

try:
    from prometheus_client import Counter, Gauge, Histogram

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
METRICS_REGISTERED = False


@dataclass(frozen=True)
class FileDownloadMetric:
    file_name: str
    file_size_bytes: int
    download_time_secs: float
    download_speed_mb_s: float


@dataclass(frozen=True)
class TrussTransferStats:
    total_manifest_size_bytes: int
    total_download_time_secs: float
    total_aggregated_mb_s: Optional[float]
    file_downloads: list[FileDownloadMetric]
    b10fs_read_speed_mbps: Optional[float]
    b10fs_decision_to_use: bool
    b10fs_enabled: bool
    b10fs_hot_starts_files: int
    b10fs_hot_starts_bytes: int
    b10fs_cold_starts_files: int
    b10fs_cold_starts_bytes: int
    success: bool
    timestamp: int

    @classmethod
    def from_json_file(cls, path: Path) -> Optional["TrussTransferStats"]:
        if not path.exists():
            return None
        try:
            with open(path) as f:
                data = json.load(f)
            file_downloads = [
                FileDownloadMetric(**fd) for fd in data.get("file_downloads", [])
            ]
            return cls(
                total_manifest_size_bytes=data["total_manifest_size_bytes"],
                total_download_time_secs=data["total_download_time_secs"],
                total_aggregated_mb_s=data.get("total_aggregated_mb_s"),
                file_downloads=file_downloads,
                b10fs_read_speed_mbps=data.get("b10fs_read_speed_mbps"),
                b10fs_decision_to_use=data["b10fs_decision_to_use"],
                b10fs_enabled=data["b10fs_enabled"],
                b10fs_hot_starts_files=data["b10fs_hot_starts_files"],
                b10fs_hot_starts_bytes=data["b10fs_hot_starts_bytes"],
                b10fs_cold_starts_files=data["b10fs_cold_starts_files"],
                b10fs_cold_starts_bytes=data["b10fs_cold_starts_bytes"],
                success=data["success"],
                timestamp=data["timestamp"],
            )
        except Exception:
            return None

    def publish_to_prometheus(self, hidden_time: float = 0.0):
        """Publish transfer stats to Prometheus metrics. Only runs once."""
        if not PROMETHEUS_AVAILABLE:
            return
        global METRICS_REGISTERED

        if METRICS_REGISTERED:
            logging.info(
                "Model cache metrics already registered, skipping."
            )  # this should never happen
            return
        else:
            # Ensure metrics are only registered once
            METRICS_REGISTERED = True

        # Define metrics with model_cache prefix
        manifest_size_gauge = Gauge(
            "model_cache_manifest_size_bytes", "Total manifest size in bytes"
        )
        # histograms have intentially wide buckets to capture a variety of download times
        download_time_histogram = Histogram(
            "model_cache_download_time_seconds",
            "Total download time in seconds",
            buckets=[0]
            + [
                2**i
                for i in range(-3, 11)  # = [0.125, .. 2048] seconds
            ]
            + [float("inf")],
        )
        download_speed_gauge = Gauge(
            "model_cache_download_speed_mbps", "Aggregated download speed in MB/s"
        )

        # File download metrics (aggregated)
        files_downloaded_counter = Counter(
            "model_cache_files_downloaded_total", "Total number of files downloaded"
        )
        total_file_size_counter = Counter(
            "model_cache_file_size_bytes_total",
            "Total size of downloaded files in bytes",
        )
        file_download_hidden_time_gauge = Gauge(
            "model_cache_file_download_hidden_time_seconds",
            "Total time hidden from user by starting the import before user code (seconds)",
        )
        file_download_time_histogram = Histogram(
            "model_cache_file_download_time_seconds",
            "File download time distribution",
            buckets=[0]
            + [
                2**i
                for i in range(-3, 11)  # = [0.125, .. 2048] seconds
            ]
            + [float("inf")],
        )
        file_download_speed_histogram = Histogram(
            "model_cache_file_download_speed_mbps",
            "File download speed distribution",
            buckets=[0]
            + [
                2**i
                for i in range(-1, 12)  # = [0.5, .. 4096] MB/s
            ]
            + [float("inf")],
        )

        # B10FS specific metrics
        b10fs_enabled_gauge = Gauge(
            "model_cache_b10fs_enabled", "Whether B10FS is enabled"
        )
        b10fs_decision_gauge = Gauge(
            "model_cache_b10fs_decision_to_use", "Whether B10FS was chosen for use"
        )
        b10fs_read_speed_gauge = Gauge(
            "model_cache_b10fs_read_speed_mbps", "B10FS read speed in Mbps"
        )
        b10fs_hot_files_gauge = Gauge(
            "model_cache_b10fs_hot_starts_files", "Number of hot start files"
        )
        b10fs_hot_bytes_gauge = Gauge(
            "model_cache_b10fs_hot_starts_bytes", "Number of hot start bytes"
        )
        b10fs_cold_files_gauge = Gauge(
            "model_cache_b10fs_cold_starts_files", "Number of cold start files"
        )
        b10fs_cold_bytes_gauge = Gauge(
            "model_cache_b10fs_cold_starts_bytes", "Number of cold start bytes"
        )

        # Transfer success metric
        transfer_success_counter = Counter(
            "model_cache_transfer_success_total",
            "Total successful transfers",
            ["success"],
        )

        # Set main transfer metrics
        manifest_size_gauge.set(self.total_manifest_size_bytes)
        download_time_histogram.observe(self.total_download_time_secs)
        file_download_hidden_time_gauge.set(hidden_time)

        if self.total_aggregated_mb_s is not None:
            download_speed_gauge.set(self.total_aggregated_mb_s)

        # Aggregate file download metrics
        total_files = len(self.file_downloads)
        total_file_bytes = sum(fd.file_size_bytes for fd in self.file_downloads)

        files_downloaded_counter.inc(total_files)
        total_file_size_counter.inc(total_file_bytes)

        # Record individual file metrics for distribution
        for fd in self.file_downloads:
            if fd.file_size_bytes > 1 * 1024 * 1024:  # Only log files larger than 1MB
                file_download_time_histogram.observe(fd.download_time_secs)
                file_download_speed_histogram.observe(fd.download_speed_mb_s)

        # B10FS metrics
        b10fs_enabled_gauge.set(1 if self.b10fs_enabled else 0)
        b10fs_decision_gauge.set(1 if self.b10fs_decision_to_use else 0)

        if self.b10fs_read_speed_mbps is not None:
            b10fs_read_speed_gauge.set(self.b10fs_read_speed_mbps)

        b10fs_hot_files_gauge.set(self.b10fs_hot_starts_files)
        b10fs_hot_bytes_gauge.set(self.b10fs_hot_starts_bytes)
        b10fs_cold_files_gauge.set(self.b10fs_cold_starts_files)
        b10fs_cold_bytes_gauge.set(self.b10fs_cold_starts_bytes)

        # Success metric
        transfer_success_counter.labels(success=str(self.success)).inc()


LAZY_DATA_RESOLVER_PATH = [
    # synced with pub static LAZY_DATA_RESOLVER_PATHS: &[&str]
    Path("/bptr/bptr-manifest"),
    Path("/bptr/bptr-manifest.json"),
    Path("/static-bptr/static-bptr-manifest.json"),
]

MISSING_COLLECTION_MESSAGE = """model_cache: Data was not collected. Missing lazy_data_resolver.block_until_download_complete().
This is a potential bug by the user implementation of model.py when using model_cache.
We need you to call the block_until_download_complete() method during __init__ or load() method of your model.
Please implement the following pattern when using model_cache.
```
import torch

class Model:
    def __init__(self, *args, **kwargs):
        self._lazy_data_resolver = kwargs["lazy_data_resolver"]

    def load():
        # work that does not require the download may be done here
        random_vector = torch.randn(1000)
        # important to collect the download before using any incomplete data
        self._lazy_data_resolver.block_until_download_complete()
        # after the call, you may use the /app/model_cache directory
        torch.load(
            "/app/model_cache/your_model.pt"
        ) * random_vector
```
"""


class LazyDataResolverV2:
    """Lazy data resolver pre-fetches data in a separate thread.
    It uses a lock to ensure that the data is only fetched once
    and that the thread is not blocked by other threads.
    """

    def __init__(self, data_dir: Path, logger: Optional[logging.Logger] = None):
        self._data_dir = data_dir
        self._lock = Lock()
        self._start_time = time.time()
        self.logger = logger or logging.getLogger(__name__)
        self._is_collected_by_user = not self.bptr_exists()
        thread = Thread(target=self._prefetch_in_thread, daemon=True)
        thread.start()

        def print_error_message_on_exit_if_not_collected():
            try:
                if not self._is_collected_by_user and thread.is_alive():
                    # if thread is still alive, and the user has not called collect,
                    # the download in flight could have been the core issue
                    self.logger.warning(
                        "An error was detected while the data was still being downloaded. "
                        + MISSING_COLLECTION_MESSAGE
                    )
            except Exception as e:
                print("Error while printing error message on exit:", e)

        atexit.register(print_error_message_on_exit_if_not_collected)

    @staticmethod
    def bptr_exists():
        """Check if the bptr manifest file exists."""
        return any(path.exists() for path in LAZY_DATA_RESOLVER_PATH)

    def _prefetch_in_thread(self):
        """Invokes the download ahead of time, before user doubles down on the download"""
        result = self.block_until_download_complete(
            log_stats=False, issue_collect=False
        )
        if not result:
            # no data to resolve, no need to collect
            self._is_collected_by_user = True
            return None
        # verify the user has called collect.
        if not self._is_collected_by_user and time.time() - self._start_time > 20:
            # issue a warning if the user has not collected after 20 seconds.
            # skip for small downloads that are less than 20 seconds
            # as the user might have a lot of work before is able to call collect.
            self.logger.warning(MISSING_COLLECTION_MESSAGE)
        time.sleep(0.5)

    @lru_cache(maxsize=None)
    def _fetch(self) -> Union[str, Exception]:
        """cached and locked method to fetch the data."""
        if not self.bptr_exists():
            return ""  # no data to resolve
        import truss_transfer

        try:
            return truss_transfer.lazy_data_resolve(str(self._data_dir))
        except Exception as e:
            self.logger.error(f"Error occurred while fetching data: {e}")
            return e

    def raise_if_not_collected(self):
        """We require the user to call `block_until_download_complete` before using the data.
        If the user has not called the method during load, we raise an error.
        """
        if not self._is_collected_by_user:
            raise RuntimeError(MISSING_COLLECTION_MESSAGE)

    def block_until_download_complete(
        self, log_stats: bool = True, issue_collect: bool = True
    ) -> str:
        """Once called, blocks until the data has been downloaded.

        example usage:
        ```
        import torch

        class Model:
            def __init__(self, *args, **kwargs):
                self._lazy_data_resolver = kwargs["lazy_data_resolver"]

            def load():
                random_vector = torch.randn(1000)
                # important to collect the download before using any incomplete data
                self._lazy_data_resolver.block_until_download_complete()
        ```

        """
        start_lock = time.time()
        publish_stats = (
            log_stats and not self._is_collected_by_user
        )  # only publish results once per resolver
        self._is_collected_by_user = issue_collect or self._is_collected_by_user
        with self._lock:
            result = self._fetch()
            if isinstance(result, Exception):
                raise RuntimeError(
                    f"Error occurred while fetching data: {result}"
                ) from result
            if log_stats and result:
                # TODO: instument the stats, which are written to /tmp/truss_transfer_stats.json
                # also add fetch time, and blocking time
                # TrussTransferStats
                fetch_t = time.time() - self._start_time
                start_lock_t = time.time() - start_lock
                stats = TrussTransferStats.from_json_file(
                    Path("/tmp/truss_transfer_stats.json")
                )
                if stats and publish_stats:
                    self.logger.info(f"model_cache: {stats}")
                    # Publish stats to Prometheus
                    stats.publish_to_prometheus()
                self.logger.info(
                    f"model_cache: Fetch took {fetch_t:.2f} seconds, of which {start_lock_t:.2f} seconds were spent blocking."
                )
            return result


if __name__ == "__main__":
    # Example usage
    print("invoking download")
    resolver = LazyDataResolverV2(Path("/example/path"))
    # similate crash
    time.sleep(0.01)
    resolver.block_until_download_complete()
    raise Exception("Simulated crash")
