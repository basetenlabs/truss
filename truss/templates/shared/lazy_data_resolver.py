from functools import cache
from pathlib import Path
from threading import Lock, Thread

LAZY_DATA_RESOLVER_PATH = Path("/bptr/bptr-manifest")


class LazyDataResolverV2:
    """Lazy data resolver pre-fetches data in a separate thread.
    It uses a lock to ensure that the data is only fetched once
    and that the thread is not blocked by other threads.
    """

    def __init__(self, data_dir: Path):
        self._data_dir = data_dir
        self._lock = Lock()

    def _prefetch_in_thread(self):
        Thread(target=self.block_until_download_complete, daemon=True).start()

    @cache
    def _fetch(self) -> str:
        if not LAZY_DATA_RESOLVER_PATH.is_file():
            return ""
        import truss_transfer

        return truss_transfer.lazy_data_resolve(str(self._data_dir))

    def block_until_download_complete(self):
        """blocks until the data has been downloaded."""
        with self._lock:
            return self._fetch()
