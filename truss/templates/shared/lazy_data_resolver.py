from functools import lru_cache
from pathlib import Path

LAZY_DATA_RESOLVER_PATH = Path("/bptr/bptr-manifest")


class LazyDataResolverV2:
    def __init__(self, data_dir: Path):
        self._data_dir = data_dir
        self.fetch()

    @lru_cache
    def fetch(self) -> str:
        if not LAZY_DATA_RESOLVER_PATH.is_file():
            return ""
        import truss_transfer

        return truss_transfer.lazy_data_resolve(str(self._data_dir))
