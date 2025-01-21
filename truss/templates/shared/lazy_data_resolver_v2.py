from pathlib import Path

try:
    import truss_transfer

    TRUSS_TRANSFER_AVAILABLE = True
except ImportError:
    TRUSS_TRANSFER_AVAILABLE = False


class LazyDataResolverV2:
    def __init__(self, data_dir: Path):
        self._data_dir: Path = data_dir

    def fetch(self):
        truss_transfer.lazy_data_resolve(str(self._data_dir))
