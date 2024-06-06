from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pydantic
import yaml
from truss.util.download import download_from_url_using_requests

LAZY_DATA_RESOLVER_PATH = Path("/bptr/bptr-manifest")


class Resolution(pydantic.BaseModel):
    url: str
    expiry: datetime


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

    def fetch(self):
        for file_name, resolved_url in self._bptr_resolution.items():
            download_from_url_using_requests(resolved_url, self._data_dir / file_name)


def _read_bptr_resolution() -> Dict[str, str]:
    if not LAZY_DATA_RESOLVER_PATH.exists():
        return {}
    bptr_manifest = BasetenPointerManifest(
        **yaml.safe_load(LAZY_DATA_RESOLVER_PATH.read_text())
    )
    resolution_map = {}
    for bptr in bptr_manifest.pointers:
        if bptr.resolution.expiry < datetime.now():
            raise RuntimeError("Baseten pointer lazy data resolution has expired")
        resolution_map[bptr.file_name] = bptr.resolution.url
    return resolution_map
