import logging
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
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

try:
    import truss_transfer

    TRUSS_TRANSFER_AVAILABLE = True
except ImportError:
    TRUSS_TRANSFER_AVAILABLE = False

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
    """Deprecation warning: This class is deprecated and will be removed in a future release.

    Please use LazyDataResolverV2 instead (using the `truss_transfer` package).
    """

    def __init__(self, data_dir: Path):
        self._data_dir: Path = data_dir
        self._bptr_resolution: Dict[str, Tuple[str, str, int]] = _read_bptr_resolution()
        self._resolution_done = False
        self._uses_b10_cache = (
            os.environ.get(BASETEN_FS_ENABLED_ENV_VAR, "False") == "True"
        )

    def cached_download_from_url_using_requests(
        self, URL: str, hash: str, file_name: str, size: int
    ):
        """Download object from URL, attempt to write to cache and symlink to data directory if applicable, data directory otherwise.
        In case of failure, write to data directory
        """
        if self._uses_b10_cache:
            file_path = CACHE_DIR / hash
            if file_path.exists():
                try:
                    os.symlink(file_path, self._data_dir / file_name)
                    return
                except FileExistsError:
                    # symlink may already exist if the inference server was restarted
                    return

        # Streaming download to keep memory usage low
        resp = requests.get(
            URL, allow_redirects=True, stream=True, timeout=BLOB_DOWNLOAD_TIMEOUT_SECS
        )
        resp.raise_for_status()

        if self._uses_b10_cache:
            try:
                # Check whether the cache has sufficient space to store the file
                cache_free_space = shutil.disk_usage(CACHE_DIR).free
                if cache_free_space < size:
                    raise OSError(
                        f"Cache directory does not have sufficient space to save file {file_name}. Free space in cache: {cache_free_space}, file size: {size}"
                    )

                file_path.parent.mkdir(parents=True, exist_ok=True)
                with file_path.open("wb") as file:
                    shutil.copyfileobj(resp.raw, file)
                # symlink to data directory
                os.symlink(file_path, self._data_dir / file_name)
                return
            except FileExistsError:
                # symlink may already exist if the inference server was restarted
                return
            except OSError as e:
                logger.debug(
                    "Failed to save artifact to cache dir, saving to data dir instead. Error: %s",
                    e,
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
            for file_name, (resolved_url, hash, size) in self._bptr_resolution.items():
                futures[
                    executor.submit(
                        self.cached_download_from_url_using_requests,
                        resolved_url,
                        hash,
                        file_name,
                        size,
                    )
                ] = file_name
            for future in as_completed(futures):
                if future.exception():
                    file_name = futures[future]
                    raise RuntimeError(f"Download failure for file {file_name}")
        self._resolution_done = True


class LazyDataResolverV2:
    def __init__(self, data_dir: Path):
        self._data_dir: Path = data_dir

    def fetch(self) -> str:
        if not LAZY_DATA_RESOLVER_PATH.is_file():
            return ""
        return truss_transfer.lazy_data_resolve(str(self._data_dir))


def _read_bptr_resolution() -> Dict[str, Tuple[str, str, int]]:
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
        resolution_map[bptr.file_name] = bptr.resolution.url, bptr.hash, bptr.size
    return resolution_map
