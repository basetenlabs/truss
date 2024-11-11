import logging
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import pydantic
import requests
import yaml

try:
    from shared.util import BLOB_DOWNLOAD_TIMEOUT_SECS
except ModuleNotFoundError:
    from truss.templates.shared.util import BLOB_DOWNLOAD_TIMEOUT_SECS

LAZY_DATA_RESOLVER_PATH = Path("/bptr/bptr-manifest")
NUM_WORKERS = 4
CACHE_DIR = Path("/cache/org/artifacts")
BASETEN_FS_ENABLED_ENV_VAR = "BASETEN_FS_ENABLED"

logger = logging.getLogger(__name__)


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
        self._bptr_resolution: Dict[str, Tuple[str, str]] = _read_bptr_resolution()
        self._resolution_done = False
        self._uses_b10_cache = (
            os.environ.get(BASETEN_FS_ENABLED_ENV_VAR, "False") == "True"
        )

    def cached_download_from_url_using_requests(
        self, URL: str, hash: str, file_name: str
    ):
        """Download object from URL, attempt to write to cache and symlink to data directory if applicable, data directory otherwise.
        In case of failure, write to data directory
        """
        if self._uses_b10_cache:
            file_path = CACHE_DIR / hash
            if file_path.exists():
                os.symlink(file_path, self._data_dir / file_name)
                return

        # Streaming download to keep memory usage low
        resp = requests.get(
            URL,
            allow_redirects=True,
            stream=True,
            timeout=BLOB_DOWNLOAD_TIMEOUT_SECS,
        )
        resp.raise_for_status()

        if self._uses_b10_cache:
            try:
                file_path.parent.mkdir(parents=True, exist_ok=True)
                with file_path.open("wb") as file:
                    shutil.copyfileobj(resp.raw, file)
                # symlink to data directory
                os.symlink(file_path, self._data_dir / file_name)
                return
            except OSError:
                logger.debug(
                    "Failed to save artifact to cache dir, saving to data dir instead"
                )
                # Cache likely has no space left on device, break to download to data dir as fallback
                pass

        file_path = self._data_dir / file_name
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("wb") as file:
            shutil.copyfileobj(resp.raw, file)

    def fetch(self):
        if self._resolution_done:
            return

        with ThreadPoolExecutor(NUM_WORKERS) as executor:
            futures = {}
            for file_name, (resolved_url, hash) in self._bptr_resolution.items():
                futures[file_name] = executor.submit(
                    self.cached_download_from_url_using_requests,
                    resolved_url,
                    hash,
                    file_name,
                )
            for file_name, future in futures.items():
                if not future:
                    raise RuntimeError(f"Download failure for file {file_name}")
        self._resolution_done = True


def _read_bptr_resolution() -> Dict[str, Tuple[str, str]]:
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
        resolution_map[bptr.file_name] = bptr.resolution.url, bptr.hash
    return resolution_map
