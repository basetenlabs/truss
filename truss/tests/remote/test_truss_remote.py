from typing import Iterator
from unittest import mock

import pytest
from requests import Response

from truss.remote.truss_remote import TrussService

TEST_SERVICE_URL = "http://test.com"


class TrussTestService(TrussService):
    def __init__(self, _service_url: str, is_draft: bool, **kwargs):
        super().__init__(_service_url, is_draft, **kwargs)

    def authenticate(self):
        return {}

    def is_live(self) -> bool:
        response = self._send_request(self._service_url, "GET")
        if response.status_code == 200:
            return True
        return False

    def is_ready(self) -> bool:
        response = self._send_request(self._service_url, "GET")
        if response.status_code == 200:
            return True
        return False

    @property
    def logs_url(self) -> str:
        raise NotImplementedError()

    @property
    def predict_url(self) -> str:
        return f"{self._service_url}/v1/models/model:predict"

    def poll_deployment_status(self, sleep_secs: int = 1) -> Iterator[str]:
        for status in ["DEPLOYING", "ACTIVE"]:
            yield status


def mock_successful_response():
    response = Response()
    response.status_code = 200
    response.json = mock.Mock(return_value={"data": {"status": "success"}})
    return response


def mock_unsuccessful_response():
    response = Response()
    response.status_code = 404
    return response


@mock.patch("requests.request", return_value=mock_successful_response())
def test_is_live(mock_request):
    service = TrussTestService(TEST_SERVICE_URL, True)
    assert service.is_live()


@mock.patch("requests.request", return_value=mock_unsuccessful_response())
def test_is_not_live(mock_request):
    service = TrussTestService(TEST_SERVICE_URL, True)
    assert service.is_live() is False


@mock.patch("requests.request", return_value=mock_successful_response())
def test_is_ready(mock_request):
    service = TrussTestService(TEST_SERVICE_URL, True)
    assert service.is_ready()


@mock.patch("requests.request", return_value=mock_unsuccessful_response())
def test_is_not_ready(mock_request):
    service = TrussTestService(TEST_SERVICE_URL, True)
    assert service.is_ready() is False


@mock.patch("requests.request", return_value=mock_successful_response())
def test_predict(mock_request):
    service = TrussTestService(TEST_SERVICE_URL, True)
    response = service.predict({"model_input": "test"})
    assert response.status_code == 200


@mock.patch("requests.request", return_value=mock_successful_response())
def test_predict_no_data(mock_request):
    service = TrussTestService(TEST_SERVICE_URL, True)
    with pytest.raises(ValueError):
        service._send_request(TEST_SERVICE_URL, "POST")
