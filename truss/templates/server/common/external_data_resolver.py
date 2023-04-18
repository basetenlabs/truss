import shutil
from pathlib import Path

import requests

BLOB_DOWNLOAD_TIMEOUT_SECS = 600  # 10 minutes


def download_external_data(data_dir: Path, config: dict):
    for item in config["external_data"]:
        backend = item["backend"]
        item_url = item["url"]
        local_data_path = item["local_data_path"]
        if backend != "http_public":
            raise ValueError(f"Unknown backend {backend}")
        _download(item_url, data_dir / local_data_path)


def _download(URL: str, download_to: Path):
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
