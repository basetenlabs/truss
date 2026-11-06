import os
from unittest.mock import Mock, patch

import pytest
import requests_mock

from truss.base.truss_config import ExternalData
from truss.util import download
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


def test_download_external_data_raises_when_b10cp_fails(tmp_path, monkeypatch):
    process = Mock()
    process.wait.return_value = 1
    monkeypatch.setenv(download.B10CP_PATH_TRUSS_ENV_VAR_NAME, "/path/to/b10cp")
    monkeypatch.setattr(
        download, "_download_from_url_using_b10cp", lambda *args: process
    )
    external_data = ExternalData(
        [{"local_data_path": "weights/model.bin", "url": TEST_DOWNLOAD_URL}]
    )

    with pytest.raises(RuntimeError, match="weights/model.bin.*exit code 1"):
        download_external_data(external_data=external_data, data_dir=tmp_path)
