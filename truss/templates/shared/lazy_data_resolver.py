from pathlib import Path

LAZY_DATA_RESOLVER_PATH = Path("/bptr/bptr-manifest")


class LazyDataResolverV2:
    def __init__(self, data_dir: Path):
        self._data_dir: Path = data_dir

    def fetch(self) -> str:
        if not LAZY_DATA_RESOLVER_PATH.is_file():
            return ""
        import truss_transfer

        return truss_transfer.lazy_data_resolve(str(self._data_dir))
