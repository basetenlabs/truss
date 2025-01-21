from pathlib import Path
from functools import lru_cache

try:
    import truss_transfer
    TRUSS_TRANSFER_AVAILABLE = True
except ImportError:
    TRUSS_TRANSFER_AVAILABLE = False

@lru_cache()
def _resolve(data_dir):
    truss_transfer.lazy_data_resolve(data_dir)

class LazyDataResolverV2:
    def __init__(self, data_dir: Path):
        self._data_dir: Path = data_dir

    def fetch(self):
        _resolve(self._data_dir)