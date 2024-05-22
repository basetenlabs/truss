from pathlib import Path

import requests
import yaml

BASETEN_POINTER_RESOLVER_PATH = Path("/btr_resolver/resolve")

# todo: define pydantic for bptr


class LazyDataResolver:
    def __init__(self, data_dir: Path):
        self._data_dir = data_dir

    def fetch(self):
        bptr_files = self._data_dir.rglob("*.bptr")
        self._bptrs = {}
        bptr_resolution = _read_btr_resolution()
        for bptr_file in bptr_files:
            bptr = yaml.safe_load(bptr_file.read_text())
            bptr_id = bptr["id"]
            if bptr_id in bptr_resolution:
                resolved_url = bptr_resolution[bptr_id]
            else:
                resolved_url = bptr["resolution"]["url"]
            _download_file(resolved_url, bptr_file.with_suffix(""))


def _read_btr_resolution():
    if not BASETEN_POINTER_RESOLVER_PATH.exists():
        return {}

    resolution = {}
    for bptr in yaml.safe_load(BASETEN_POINTER_RESOLVER_PATH.read_text()):
        # todo check expiry
        resolution[bptr["id"]] = bptr["resolution"]["url"]
    return resolution


def _download_file(url: str, to: Path):
    print(f"Downloading to {to}")
    resp = requests.get(url, stream=True)
    if resp.status_code != 200:
        raise RuntimeError(f"Unable to download {url}")

    with to.open("wb") as fp:
        for chunk in resp.iter_content(chunk_size=1024):
            if chunk:
                fp.write(chunk)
