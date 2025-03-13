import datetime
import json
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Callable
from unittest.mock import patch

import pytest
import requests_mock

from truss.templates.shared.lazy_data_resolver import (
    BASETEN_FS_ENABLED_ENV_VAR,
    LAZY_DATA_RESOLVER_PATH,
    LazyDataResolver,
)


@pytest.fixture
def baseten_pointer_manifest_mock() -> Callable:
    def _baseten_pointer_manifest_mock(
        foo_expiry_timestamp: int, bar_expiry_timestamp: int
    ):
        return f"""
pointers:
- uid: foo
  file_name: foo-name
  hashtype: hash-type
  hash: foo-hash
  size: 100
  resolution:
    url: https://foo-rl
    expiration_timestamp: {foo_expiry_timestamp}
- uid: bar
  file_name: bar-name
  hashtype: hash-type
  hash: bar-hash
  size: 1000
  resolution:
    url: https://bar-rl
    expiration_timestamp: {bar_expiry_timestamp}
"""

    return _baseten_pointer_manifest_mock


def test_lazy_data_resolution_not_found():
    ldr = LazyDataResolver(Path("foo"))
    assert not LAZY_DATA_RESOLVER_PATH.exists()
    assert ldr._bptr_resolution == {}


@pytest.mark.parametrize(
    "foo_expiry,bar_expiry,expectation",
    [
        (
            int(
                datetime.datetime(3000, 1, 1, tzinfo=datetime.timezone.utc).timestamp()
            ),
            int(
                datetime.datetime(3000, 1, 1, tzinfo=datetime.timezone.utc).timestamp()
            ),
            nullcontext(),
        ),
        (
            int(
                datetime.datetime(2020, 1, 1, tzinfo=datetime.timezone.utc).timestamp()
            ),
            int(
                datetime.datetime(2020, 1, 1, tzinfo=datetime.timezone.utc).timestamp()
            ),
            pytest.raises(RuntimeError),
        ),
    ],
)
def test_lazy_data_resolution(
    baseten_pointer_manifest_mock, foo_expiry, bar_expiry, expectation, tmp_path
):
    baseten_pointer_manifest_mock = baseten_pointer_manifest_mock(
        foo_expiry, bar_expiry
    )
    manifest_path = tmp_path / "bptr" / "bptr-manifest"
    manifest_path.parent.mkdir()
    manifest_path.touch()
    manifest_path.write_text(baseten_pointer_manifest_mock)
    with patch(
        "truss.templates.shared.lazy_data_resolver.LAZY_DATA_RESOLVER_PATH",
        manifest_path,
    ):
        with expectation:
            ldr = LazyDataResolver(Path("foo"))
            assert ldr._bptr_resolution == {
                "foo-name": ("https://foo-rl", "foo-hash", 100),
                "bar-name": ("https://bar-rl", "bar-hash", 1000),
            }


@pytest.mark.parametrize(
    "foo_expiry,bar_expiry",
    [
        (
            int(
                datetime.datetime(3000, 1, 1, tzinfo=datetime.timezone.utc).timestamp()
            ),
            int(
                datetime.datetime(3000, 1, 1, tzinfo=datetime.timezone.utc).timestamp()
            ),
        )
    ],
)
def test_lazy_data_fetch(
    baseten_pointer_manifest_mock, foo_expiry, bar_expiry, tmp_path
):
    baseten_pointer_manifest_mock = baseten_pointer_manifest_mock(
        foo_expiry, bar_expiry
    )
    manifest_path = tmp_path / "bptr" / "bptr-manifest"
    manifest_path.parent.mkdir()
    manifest_path.touch()
    manifest_path.write_text(baseten_pointer_manifest_mock)
    with patch(
        "truss.templates.shared.lazy_data_resolver.LAZY_DATA_RESOLVER_PATH",
        manifest_path,
    ):
        data_dir = Path(tmp_path)
        ldr = LazyDataResolver(data_dir)
        with requests_mock.Mocker() as m:
            for file_name, (url, _, _) in ldr._bptr_resolution.items():
                resp = {"file_name": file_name, "url": url}
                m.get(url, json=resp)
            ldr.fetch()
            for file_name, (url, _, _) in ldr._bptr_resolution.items():
                assert (ldr._data_dir / file_name).read_text() == json.dumps(
                    {"file_name": file_name, "url": url}
                )


@pytest.mark.parametrize(
    "foo_expiry,bar_expiry",
    [
        (
            int(
                datetime.datetime(3000, 1, 1, tzinfo=datetime.timezone.utc).timestamp()
            ),
            int(
                datetime.datetime(3000, 1, 1, tzinfo=datetime.timezone.utc).timestamp()
            ),
        )
    ],
)
def test_lazy_data_fetch_to_cache_non_200_status(
    baseten_pointer_manifest_mock, foo_expiry, bar_expiry, tmp_path, monkeypatch
):
    monkeypatch.setenv(BASETEN_FS_ENABLED_ENV_VAR, "True")
    baseten_pointer_manifest_mock = baseten_pointer_manifest_mock(
        foo_expiry, bar_expiry
    )
    manifest_path = tmp_path / "bptr" / "bptr-manifest"
    manifest_path.parent.mkdir()
    manifest_path.touch()
    manifest_path.write_text(baseten_pointer_manifest_mock)
    cache_dir = tmp_path / "cache" / "org" / "artifacts"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.touch()
    with (
        patch(
            "truss.templates.shared.lazy_data_resolver.LAZY_DATA_RESOLVER_PATH",
            manifest_path,
        ) as _,
        patch("truss.templates.shared.lazy_data_resolver.CACHE_DIR", cache_dir) as _,
    ):
        data_dir = Path(tmp_path)
        ldr = LazyDataResolver(data_dir)
        assert ldr._uses_b10_cache
        with requests_mock.Mocker() as m:
            for _, (url, _, _) in ldr._bptr_resolution.items():
                m.get(url, status_code=500)
            with pytest.raises(RuntimeError):
                ldr.fetch()


@pytest.mark.parametrize(
    "foo_expiry,bar_expiry",
    [
        (
            int(
                datetime.datetime(3000, 1, 1, tzinfo=datetime.timezone.utc).timestamp()
            ),
            int(
                datetime.datetime(3000, 1, 1, tzinfo=datetime.timezone.utc).timestamp()
            ),
        )
    ],
)
def test_lazy_data_fetch_to_cache(
    baseten_pointer_manifest_mock, foo_expiry, bar_expiry, tmp_path, monkeypatch
):
    monkeypatch.setenv(BASETEN_FS_ENABLED_ENV_VAR, "True")
    baseten_pointer_manifest_mock = baseten_pointer_manifest_mock(
        foo_expiry, bar_expiry
    )
    manifest_path = tmp_path / "bptr" / "bptr-manifest"
    manifest_path.parent.mkdir()
    manifest_path.touch()
    manifest_path.write_text(baseten_pointer_manifest_mock)
    cache_dir = tmp_path / "cache" / "org" / "artifacts"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.touch()
    with (
        patch(
            "truss.templates.shared.lazy_data_resolver.LAZY_DATA_RESOLVER_PATH",
            manifest_path,
        ) as _,
        patch(
            "truss.templates.shared.lazy_data_resolver.CACHE_DIR", cache_dir
        ) as CACHE_DIR,
    ):
        data_dir = Path(tmp_path)
        ldr = LazyDataResolver(data_dir)
        assert ldr._uses_b10_cache
        with requests_mock.Mocker() as m:
            for file_name, (url, hash, _) in ldr._bptr_resolution.items():
                resp = {"file_name": file_name, "url": url}
                m.get(url, json=resp)
            ldr.fetch()
            for file_name, (url, hash, _) in ldr._bptr_resolution.items():
                assert (CACHE_DIR / hash).read_text() == json.dumps(
                    {"file_name": file_name, "url": url}
                )
                assert os.path.islink(ldr._data_dir / file_name)
                assert os.readlink(ldr._data_dir / file_name) == str(CACHE_DIR / hash)


@pytest.mark.parametrize(
    "foo_expiry,bar_expiry",
    [
        (
            int(
                datetime.datetime(3000, 1, 1, tzinfo=datetime.timezone.utc).timestamp()
            ),
            int(
                datetime.datetime(3000, 1, 1, tzinfo=datetime.timezone.utc).timestamp()
            ),
        )
    ],
)
def test_lazy_data_fetch_to_cache_fallback_if_no_space(
    baseten_pointer_manifest_mock, foo_expiry, bar_expiry, tmp_path, monkeypatch
):
    monkeypatch.setenv(BASETEN_FS_ENABLED_ENV_VAR, "True")
    baseten_pointer_manifest_mock = baseten_pointer_manifest_mock(
        foo_expiry, bar_expiry
    )
    manifest_path = tmp_path / "bptr" / "bptr-manifest"
    manifest_path.parent.mkdir()
    manifest_path.touch()
    manifest_path.write_text(baseten_pointer_manifest_mock)
    cache_dir = tmp_path / "cache" / "org" / "artifacts"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.touch()
    with (
        patch(
            "truss.templates.shared.lazy_data_resolver.LAZY_DATA_RESOLVER_PATH",
            manifest_path,
        ) as _,
        patch("truss.templates.shared.lazy_data_resolver.CACHE_DIR", cache_dir) as _,
        patch(
            "truss.templates.shared.lazy_data_resolver.shutil.disk_usage"
        ) as mock_disk_usage,
    ):
        data_dir = Path(tmp_path)
        ldr = LazyDataResolver(data_dir)
        assert ldr._uses_b10_cache

        mock_disk_usage.return_value.free = 1
        with requests_mock.Mocker() as m:
            for file_name, (url, _, _) in ldr._bptr_resolution.items():
                resp = {"file_name": file_name, "url": url}
                m.get(url, json=resp)
            ldr.fetch()

            for file_name, (url, _, _) in ldr._bptr_resolution.items():
                assert (ldr._data_dir / file_name).read_text() == json.dumps(
                    {"file_name": file_name, "url": url}
                )


@pytest.mark.parametrize(
    "foo_expiry,bar_expiry",
    [
        (
            int(
                datetime.datetime(3000, 1, 1, tzinfo=datetime.timezone.utc).timestamp()
            ),
            int(
                datetime.datetime(3000, 1, 1, tzinfo=datetime.timezone.utc).timestamp()
            ),
        )
    ],
)
def test_lazy_data_fetch_cached(
    baseten_pointer_manifest_mock, foo_expiry, bar_expiry, tmp_path, monkeypatch
):
    monkeypatch.setenv(BASETEN_FS_ENABLED_ENV_VAR, "True")
    baseten_pointer_manifest_mock = baseten_pointer_manifest_mock(
        foo_expiry, bar_expiry
    )
    manifest_path = tmp_path / "bptr" / "bptr-manifest"
    manifest_path.parent.mkdir()
    manifest_path.touch()
    manifest_path.write_text(baseten_pointer_manifest_mock)
    cache_dir = tmp_path / "cache" / "org" / "artifacts"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.touch()
    with (
        patch(
            "truss.templates.shared.lazy_data_resolver.LAZY_DATA_RESOLVER_PATH",
            manifest_path,
        ) as _,
        patch(
            "truss.templates.shared.lazy_data_resolver.CACHE_DIR", cache_dir
        ) as CACHE_DIR,
    ):
        data_dir = Path(tmp_path)
        ldr = LazyDataResolver(data_dir)
        assert ldr._uses_b10_cache
        with requests_mock.Mocker() as m:
            for file_name, (url, hash, _) in ldr._bptr_resolution.items():
                resp = {"file_name": file_name, "url": url}
                (CACHE_DIR / hash).write_text(json.dumps(resp))
                m.get(url, json=resp)
            ldr.fetch()
            for file_name, (url, hash, _) in ldr._bptr_resolution.items():
                assert (CACHE_DIR / hash).read_text() == json.dumps(
                    {"file_name": file_name, "url": url}
                )
                assert not m.called
                assert os.path.islink(ldr._data_dir / file_name)
                assert os.readlink(ldr._data_dir / file_name) == str(CACHE_DIR / hash)
