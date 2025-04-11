import logging
import time
from functools import cache
from pathlib import Path
from threading import Lock, Thread
from typing import Optional

LAZY_DATA_RESOLVER_PATH = Path("/bptr/bptr-manifest")

MISSING_COLLECTION_MESSAGE = """LazyDataResolverV2: Data was not collected, using block_until_download_complete(). This is a bug by the users implementation of Model.
Please implement the following pattern.
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
        self._is_collected = False
        Thread(target=self._prefetch_in_thread, daemon=True).start()

    def _prefetch_in_thread(self):
        result = self.block_until_download_complete(
            log_stats=False, issue_collect=False
        )
        if not result:
            # no data to resolve
            return None
        # verify the user has called collect.
        if not self._is_collected and time.time() - self._start_time > 15:
            # issue a warning if the user has not collected after 15 seconds.
            # skip if its less than 15seconds, as the user might have a lot of work before is abled to call collect.
            self.logger.warning(MISSING_COLLECTION_MESSAGE)
        time.sleep(75)
        if not self._is_collected:
            # issue an error message, if the user has not collected after 90 seconds
            self.logger.error(MISSING_COLLECTION_MESSAGE)

    @cache
    def _fetch(self) -> str:
        if not LAZY_DATA_RESOLVER_PATH.is_file():
            return ""  # no data to resolve
        import truss_transfer

        return truss_transfer.lazy_data_resolve(str(self._data_dir))

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
        self._is_collected = issue_collect or self._is_collected
        with self._lock:
            result = self._fetch()
            if log_stats and result:
                self.logger.info(
                    f"LazyDataResolverV2: Fetch took {time.time() - self._start_time:.2f} seconds, of which {time.time() - start_lock:.2f} seconds were spent blocking."
                )
            return result
