import atexit
import logging
import time
from functools import lru_cache
from pathlib import Path
from threading import Lock, Thread
from typing import Optional

LAZY_DATA_RESOLVER_PATH = Path("/bptr/bptr-manifest")

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
        self._is_collected_by_user = False
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
    def _fetch(self) -> str:
        """cached and locked method to fetch the data."""
        if not LAZY_DATA_RESOLVER_PATH.is_file():
            return ""  # no data to resolve
        import truss_transfer

        return truss_transfer.lazy_data_resolve(str(self._data_dir))

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
        self._is_collected_by_user = issue_collect or self._is_collected_by_user
        with self._lock:
            result = self._fetch()
            if log_stats and result:
                self.logger.info(
                    f"model_cache: Fetch took {time.time() - self._start_time:.2f} seconds, of which {time.time() - start_lock:.2f} seconds were spent blocking."
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
