import os
from unittest.mock import patch

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
