from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import pydantic
import yaml

try:
    from shared.util import download_from_url_using_requests
except ModuleNotFoundError:
    from truss.templates.shared.util import download_from_url_using_requests

LAZY_DATA_RESOLVER_PATH = Path("/bptr/bptr-manifest")
NUM_WORKERS = 4


class Resolution(pydantic.BaseModel):
    url: str
    expiration_timestamp: int


class BasetenPointer(pydantic.BaseModel):
    """Specification for lazy data resolution for download of large files, similar to Git LFS pointers"""

    resolution: Resolution
    uid: str
    file_name: str
    hashtype: str
    hash: str
    size: int


class BasetenPointerManifest(pydantic.BaseModel):
    pointers: List[BasetenPointer]


class LazyDataResolver:
    def __init__(self, data_dir: Path):
        self._data_dir: Path = data_dir
        self._bptr_resolution: Dict[str, str] = _read_bptr_resolution()
        self._resolution_done = False

    def fetch(self):
        if self._resolution_done:
            return

        with ThreadPoolExecutor(NUM_WORKERS) as executor:
            futures = {}
            for file_name, resolved_url in self._bptr_resolution.items():
                futures[file_name] = executor.submit(
                    download_from_url_using_requests,
                    resolved_url,
                    self._data_dir / file_name,
                )
            for file_name, future in futures.items():
                if not future:
                    raise RuntimeError(f"Download failure for file {file_name}")
        self._resolution_done = True


def _read_bptr_resolution() -> Dict[str, str]:
    if not LAZY_DATA_RESOLVER_PATH.is_file():
        return {}
    bptr_manifest = BasetenPointerManifest(
        **yaml.safe_load(LAZY_DATA_RESOLVER_PATH.read_text())
    )
    resolution_map = {}
    for bptr in bptr_manifest.pointers:
        if bptr.resolution.expiration_timestamp < int(
            datetime.now(timezone.utc).timestamp()
        ):
            raise RuntimeError("Baseten pointer lazy data resolution has expired")
        resolution_map[bptr.file_name] = bptr.resolution.url
    return resolution_map
