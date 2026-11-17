import os
from unittest.mock import patch

import pytest
import requests_mock

from truss.base.truss_config import ExternalData
from truss.util.download import download_external_data

TEST_DOWNLOAD_URL = "http://example.com/some-download-url"


def test_download(tmp_path):
    mocked_download_content = b"mocked content"
    with patch.dict(os.environ, {}), requests_mock.Mocker() as m:
        m.get(TEST_DOWNLOAD_URL, content=mocked_download_content)
        external_data = ExternalData(
            [{"local_data_path": "foo", "url": TEST_DOWNLOAD_URL}]
        )
        download_external_data(external_data=external_data, data_dir=tmp_path)

        with open(tmp_path / "foo", "rb") as f:
            content = f.read()

        assert content == mocked_download_content


def test_download_into_nested_subdir(tmp_path):
    mocked_download_content = b"mocked content"
    with patch.dict(os.environ, {}), requests_mock.Mocker() as m:
        m.get(TEST_DOWNLOAD_URL, content=mocked_download_content)
        external_data = ExternalData(
            [{"local_data_path": "foo/bar/baz", "url": TEST_DOWNLOAD_URL}]
        )
        download_external_data(external_data=external_data, data_dir=tmp_path)

        with open(tmp_path / "foo" / "bar" / "baz", "rb") as f:
            content = f.read()

        assert content == mocked_download_content


@pytest.mark.parametrize("local_data_path", ["foo/../../escaped", "/abs/escaped"])
def test_download_rejects_path_outside_data_dir(tmp_path, local_data_path):
    data_dir = tmp_path / "data"
    with patch.dict(os.environ, {}), requests_mock.Mocker() as m:
        m.get(TEST_DOWNLOAD_URL, content=b"pwned")
        external_data = ExternalData(
            [{"local_data_path": local_data_path, "url": TEST_DOWNLOAD_URL}]
        )
        with pytest.raises(ValueError):
            download_external_data(external_data=external_data, data_dir=data_dir)
        assert not m.called
    assert not (tmp_path / "escaped").exists()


def test_download_dotdot_after_symlink_stays_in_data_dir(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    try:
        (data_dir / "alias").symlink_to(data_dir, target_is_directory=True)
    except OSError:
        # Creating symlinks on Windows requires extra privileges.
        pytest.skip("symlinks not supported")
    with patch.dict(os.environ, {}), requests_mock.Mocker() as m:
        m.get(TEST_DOWNLOAD_URL, content=b"ok")
        external_data = ExternalData(
            [{"local_data_path": "alias/../escaped", "url": TEST_DOWNLOAD_URL}]
        )
        download_external_data(external_data=external_data, data_dir=data_dir)
    # Downloaded to the lexically checked path, not through the symlink.
    assert (data_dir / "escaped").read_bytes() == b"ok"
    assert not (tmp_path / "escaped").exists()


def test_download_allows_dotdot_staying_inside_data_dir(tmp_path):
    with patch.dict(os.environ, {}), requests_mock.Mocker() as m:
        m.get(TEST_DOWNLOAD_URL, content=b"ok")
        external_data = ExternalData(
            [{"local_data_path": "foo/../bar", "url": TEST_DOWNLOAD_URL}]
        )
        download_external_data(external_data=external_data, data_dir=tmp_path)
    assert (tmp_path / "bar").read_bytes() == b"ok"
