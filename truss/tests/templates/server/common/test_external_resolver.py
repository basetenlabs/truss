from unittest.mock import patch

import pytest
from truss.templates.server.common.external_data_resolver import (
    _external_data_items,
    download_external_data,
)
from truss.truss_config import ExternalData, ExternalDataItem, TrussConfig


def test_external_data_items(tmp_path):
    url1 = "http://dummy1"
    url2 = "http://dummy2"
    data_dir = tmp_path
    config = TrussConfig(
        external_data=ExternalData(
            items=[
                ExternalDataItem(url=url1, local_data_path="dummy1"),
                ExternalDataItem(url=url2, local_data_path="dir/dummy2"),
            ]
        )
    )

    items = _external_data_items(data_dir, config.to_dict())
    assert items == [
        (url1, tmp_path / "dummy1"),
        (url2, tmp_path / "dir" / "dummy2"),
    ]


def test_external_data_items_unsupported_backend(tmp_path):
    url = "http://dummy1"
    data_dir = tmp_path
    config = TrussConfig(
        external_data=ExternalData(
            items=[
                ExternalDataItem(
                    url=url, local_data_path="dummy1", backend="unsupported"
                ),
            ]
        )
    )

    with pytest.raises(ValueError):
        _external_data_items(data_dir, config.to_dict())


def test_external_data_items_empty(tmp_path):
    data_dir = tmp_path
    config = TrussConfig()

    items = _external_data_items(data_dir, config.to_dict())
    assert items == []


@patch("truss.templates.server.common.external_data_resolver._try_b10cp")
@patch("truss.templates.server.common.external_data_resolver._download_using_requests")
def test_download_external_data(
    _try_b10cp_mock, _download_using_requests_mock, tmp_path
):
    _try_b10cp_mock.return_value = False
    url = "http://dummy1"
    data_dir = tmp_path
    config = TrussConfig(
        external_data=ExternalData(
            items=[
                ExternalDataItem(url=url, local_data_path="dummy1"),
            ]
        )
    )
    download_external_data(data_dir, config.to_dict())
    assert _try_b10cp_mock.called_with(url, tmp_path / "dummy1")
    assert _download_using_requests_mock.called_with(url, tmp_path / "dummy1")
