import json
import tempfile
import time
from pathlib import Path

import pytest

from truss.templates.shared.lazy_data_resolver import (
    TRUSS_TRANSFER_AVAILABLE,
    LazyDataResolverV2,
)

LAZY_DATA_RESOLVER_PATH = Path("/bptr/bptr-manifest")
TARGET_FILE = Path("nested/config.json")


def write_bptr_manifest_to_file(expiration_timestamp: int = 2683764059):
    bptr_manifest = {
        "pointers": [
            {
                "resolution": {
                    "url": "https://raw.githubusercontent.com/basetenlabs/truss/00e01b679afbe353b0b2fe4de6b138d912bb7167/.circleci/config.yml",
                    "expiration_timestamp": expiration_timestamp,
                },
                "uid": "8c6b2f215f0333437cdc3fe7c79be0c802847d2f2a0ccdc0bb251814e63cf375",
                "file_name": TARGET_FILE.as_posix(),
                "hashtype": "blake3",
                "hash": "8c6b2f215f0333437cdc3fe7c79be0c802847d2f2a0ccdc0bb251814e63cf375",
                "size": 1482,
            }
        ]
    }
    # write to LAZY_DATA_RESOLVER_PATH
    with open(LAZY_DATA_RESOLVER_PATH, "w") as f:
        json.dump(bptr_manifest, f)


@pytest.mark.skipif(not TRUSS_TRANSFER_AVAILABLE, reason="Truss Transfer not available")
def test_lazy_data_resolver_v2():
    # truss_transfer reads from LAZY_DATA_RESOLVER_PATH
    if LAZY_DATA_RESOLVER_PATH.exists():
        LAZY_DATA_RESOLVER_PATH.unlink()
    with pytest.raises(Exception):
        # LAZY_DATA_RESOLVER_PATH does not exist
        # should raise an exception
        LazyDataResolverV2(Path("/tmp")).fetch()

    try:
        LAZY_DATA_RESOLVER_PATH.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        pytest.skip(
            f"Unable to create {LAZY_DATA_RESOLVER_PATH} due to missing os permissions: {e}"
        )

    # without LAZY_DATA_RESOLVER_PATH -> does not create folder / file
    with tempfile.TemporaryDirectory() as tempdir:
        data_dir = Path(tempdir)
        resolver = LazyDataResolverV2(data_dir).fetch()
        assert not (data_dir / TARGET_FILE).exists()

    # with LAZY_DATA_RESOLVER_PATH -> fetches data
    with tempfile.TemporaryDirectory() as tempdir:
        data_dir = Path(tempdir)
        write_bptr_manifest_to_file()
        resolver = LazyDataResolverV2(data_dir).fetch()
        resolver.fetch()
        assert (data_dir / TARGET_FILE).exists()
        assert (data_dir / TARGET_FILE).stat().st_size == 1482

    # with expired LAZY_DATA_RESOLVER_PATH -> raises exception
    with tempfile.TemporaryDirectory() as tempdir:
        data_dir = Path(tempdir)
        write_bptr_manifest_to_file(expiration_timestamp=int(time.time()) - 1)
        resolver = LazyDataResolverV2(data_dir).fetch()
        with pytest.raises(Exception):
            resolver.fetch()


if __name__ == "__main__":
    test_lazy_data_resolver_v2()
