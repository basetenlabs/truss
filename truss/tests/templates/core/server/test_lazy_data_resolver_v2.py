import json
import os
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

import pytest

from truss.templates.shared.lazy_data_resolver import (
    TRUSS_TRANSFER_AVAILABLE,
    LazyDataResolverV2,
)

LAZY_DATA_RESOLVER_PATH = Path("/bptr/bptr-manifest")
TARGET_FILE = Path("nested/config.json")


def write_bptr_manifest_to_file(
    expiration_timestamp: int = 2683764059, num_files: int = 1
):
    bptr_manifest = {
        "pointers": [
            {
                "resolution": {
                    "url": "https://raw.githubusercontent.com/basetenlabs/truss/00e01b679afbe353b0b2fe4de6b138d912bb7167/.circleci/config.yml",
                    "expiration_timestamp": expiration_timestamp,
                },
                "uid": "8c6b2f215f0333437cdc3fe7c79be0c802847d2f2a0ccdc0bb251814e63cf375",
                "file_name": TARGET_FILE.as_posix()
                if i == 0
                else str(i) + TARGET_FILE.as_posix(),
                "hashtype": "blake3",
                "hash": "8c6b2f215f0333437cdc3fe7c79be0c802847d2f2a0ccdc0bb251814e63cf375",
                "size": 1482,
            }
            for i in range(num_files)
        ]
    }
    # write to LAZY_DATA_RESOLVER_PATH
    with open(LAZY_DATA_RESOLVER_PATH, "w") as f:
        json.dump(bptr_manifest, f)
        print(bptr_manifest)


def write_bptr_manifest_to_file_invalid_json():
    with open(LAZY_DATA_RESOLVER_PATH, "w") as f:
        json.dump({"invalid": True}, f)


@contextmanager
def setup_v2_bptr():
    """Context manager that sets up and tears down the BPTR environment."""
    # Clean up any existing file
    if LAZY_DATA_RESOLVER_PATH.exists():
        LAZY_DATA_RESOLVER_PATH.unlink()

    try:
        # Verify that accessing non-existent file raises exception
        with pytest.raises(Exception):
            LazyDataResolverV2(Path("/tmp")).fetch()

        # Create parent directory
        try:
            LAZY_DATA_RESOLVER_PATH.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(
                f"Unable to create {LAZY_DATA_RESOLVER_PATH} due to missing os permissions: {e}"
            )
            if sys.platform.startswith("win"):
                pytest.mark.skip(reason="Windows does not designed for running bptr")
            else:
                raise e

        yield  # Let the caller use the setup environment

    finally:
        # Clean up after tests
        if LAZY_DATA_RESOLVER_PATH.exists():
            LAZY_DATA_RESOLVER_PATH.unlink()


@pytest.mark.skipif(not TRUSS_TRANSFER_AVAILABLE, reason="Truss Transfer not available")
def test_lazy_data_resolver_v2_invalid():
    with setup_v2_bptr():
        # with invalid data
        write_bptr_manifest_to_file_invalid_json()

        with tempfile.TemporaryDirectory() as tempdir:
            data_dir = Path(tempdir)
            resolver = LazyDataResolverV2(data_dir)
            with pytest.raises(Exception):
                resolver.fetch()
                assert not (data_dir / TARGET_FILE).exists()


@pytest.mark.skipif(not TRUSS_TRANSFER_AVAILABLE, reason="Truss Transfer not available")
def test_lazy_data_resolver_v2_regular():
    with setup_v2_bptr():
        # with LAZY_DATA_RESOLVER_PATH -> fetches data
        with tempfile.TemporaryDirectory() as tempdir:
            data_dir = Path(tempdir)
            write_bptr_manifest_to_file()
            resolver = LazyDataResolverV2(data_dir)
            resolver.fetch()
            assert (data_dir / TARGET_FILE).exists()
            assert (data_dir / TARGET_FILE).stat().st_size == 1482


@pytest.mark.skipif(not TRUSS_TRANSFER_AVAILABLE, reason="Truss Transfer not available")
def test_lazy_data_resolver_v2_regular_baseten_fs():
    with setup_v2_bptr():
        old_os_setting = os.environ.get("BASETEN_FS_ENABLED")
        # with LAZY_DATA_RESOLVER_PATH -> fetches data
        try:
            os.environ["BASETEN_FS_ENABLED"] = "True"
            with tempfile.TemporaryDirectory() as tempdir:
                data_dir = Path(tempdir)
                write_bptr_manifest_to_file()
                resolver = LazyDataResolverV2(data_dir)
                resolver.fetch()
                assert (data_dir / TARGET_FILE).exists()
                assert (data_dir / TARGET_FILE).stat().st_size == 1482
        finally:
            if old_os_setting:
                os.environ["BASETEN_FS_ENABLED"] = old_os_setting
            else:
                del os.environ["BASETEN_FS_ENABLED"]


@pytest.mark.skipif(not TRUSS_TRANSFER_AVAILABLE, reason="Truss Transfer not available")
def test_lazy_data_resolver_v2_multiple_files():
    with setup_v2_bptr():
        # with multiple files
        with tempfile.TemporaryDirectory() as tempdir:
            data_dir = Path(tempdir)
            write_bptr_manifest_to_file(num_files=2)
            resolver = LazyDataResolverV2(data_dir)
            resolver.fetch()
            assert (data_dir / TARGET_FILE).exists()
            assert (data_dir / TARGET_FILE).stat().st_size == 1482


@pytest.mark.skipif(not TRUSS_TRANSFER_AVAILABLE, reason="Truss Transfer not available")
def test_lazy_data_resolver_v2_expired():
    with setup_v2_bptr():
        # with expired LAZY_DATA_RESOLVER_PATH -> raises exception
        with tempfile.TemporaryDirectory() as tempdir:
            data_dir = Path(tempdir)
            write_bptr_manifest_to_file(expiration_timestamp=int(time.time()) - 1)
            resolver = LazyDataResolverV2(data_dir)
            with pytest.raises(Exception):
                resolver.fetch()
