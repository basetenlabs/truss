import shutil
import subprocess
from pathlib import Path

import requests

BLOB_DOWNLOAD_TIMEOUT_SECS = 600  # 10 minutes
B10CP_PATH = "/app/bin/b10cp"


def download_external_data(data_dir: Path, config: dict):
    if "external_data" not in config:
        return

    for item in config["external_data"]:
        backend = item.get("backend", "http_public")
        item_url = item["url"]
        local_data_path = item["local_data_path"]
        if backend != "http_public":
            raise ValueError(f"Unknown backend {backend}")
        _download(item_url, data_dir / local_data_path)


def _download(URL: str, download_to: Path):
    download_to.parent.mkdir(exist_ok=True, parents=True)

    b10cp_success = _try_b10cp(URL, download_to)
    if b10cp_success:
        return

    # Fallback if b10cp can't handle the url
    # Streaming download to keep memory usage low
    resp = requests.get(
        URL,
        allow_redirects=True,
        stream=True,
        timeout=BLOB_DOWNLOAD_TIMEOUT_SECS,
    )
    resp.raise_for_status()
    with download_to.open("wb") as file:
        shutil.copyfileobj(resp.raw, file)


def _try_b10cp(URL: str, download_to: Path) -> bool:
    if not Path(B10CP_PATH).exists():
        return False

    proc = subprocess.run(
        [
            B10CP_PATH,
            "-source",
            URL,  # Add quotes to work with any special characters.
            "-target",
            str(download_to),
        ],
        check=False,
    )
    return proc.returncode == 0
