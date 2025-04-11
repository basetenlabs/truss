from functools import lru_cache
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
        Thread(target=self._prefetch_thread, daemon=True).start()

    def _prefetch_thread(self):
        self._fetch()

    @lru_cache
    def _fetch(self) -> str:
        if not LAZY_DATA_RESOLVER_PATH.is_file():
            return ""
        import truss_transfer

        return truss_transfer.lazy_data_resolve(str(self._data_dir))

    def block_until_fetched(self):
        """blocks until the data has been downloaded."""
        with self._lock:
            return self._fetch()

    def fetch(self) -> str:
        """deprecated: use block_until_fetched instead"""
        return self.block_until_fetched()
