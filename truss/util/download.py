import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import requests

from truss.base.truss_config import ExternalData

B10CP_EXECUTABLE_NAME = "b10cp"
BLOB_DOWNLOAD_TIMEOUT_SECS = 600  # 10 minutes
B10CP_PATH_TRUSS_ENV_VAR_NAME = "B10CP_PATH_TRUSS"


def download_external_data(external_data: Optional[ExternalData], data_dir: Path):
    if external_data is None:
        return
    data_dir.mkdir(exist_ok=True)
    b10cp_path = _b10cp_path()

    # Normalize lexically so `..` can't escape data_dir, and download to exactly the
    # checked path. Symlinks are deliberately not resolved, so symlinked
    # subdirectories keep working, but `..` after a symlink can't escape either.
    abs_data_dir = Path(os.path.abspath(data_dir))
    downloads: list[tuple[str, Path]] = []
    for item in external_data.items:
        path = Path(os.path.abspath(data_dir / item.local_data_path))
        if abs_data_dir not in path.parents:
            raise ValueError(
                "Local data path of external data cannot point to outside data directory"
            )
        path.parent.mkdir(exist_ok=True, parents=True)
        downloads.append((item.url, path))

    if b10cp_path is not None:
        print("b10cp found, using it to download external data")
        _download_external_data_using_b10cp(b10cp_path, downloads)
        return

    # slow path
    _download_external_data_using_requests(downloads)


def _b10cp_path() -> Optional[str]:
    return os.environ.get(B10CP_PATH_TRUSS_ENV_VAR_NAME)


def _download_external_data_using_b10cp(
    b10cp_path: str, downloads: list[tuple[str, Path]]
):
    procs = []
    # TODO(pankaj) Limit concurrency here
    for url, path in downloads:
        proc = _download_from_url_using_b10cp(b10cp_path, url, path)
        procs.append(proc)

    for proc in procs:
        proc.wait()


def _download_from_url_using_b10cp(b10cp_path: str, url: str, download_to: Path):
    return subprocess.Popen(
        [
            b10cp_path,
            "-source",
            url,  # Add quotes to work with any special characters.
            "-target",
            str(download_to),
        ]
    )


def _download_external_data_using_requests(downloads: list[tuple[str, Path]]):
    for url, path in downloads:
        download_from_url_using_requests(url, path)


def download_from_url_using_requests(URL: str, download_to: Path):
    # Streaming download to keep memory usage low
    resp = requests.get(
        URL, allow_redirects=True, stream=True, timeout=BLOB_DOWNLOAD_TIMEOUT_SECS
    )
    resp.raise_for_status()
    with download_to.open("wb") as file:
        shutil.copyfileobj(resp.raw, file)
