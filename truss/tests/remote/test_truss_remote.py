from unittest import mock

import pytest
from requests import Response
from truss.remote.truss_remote import TrussService

TEST_SERVICE_URL = "http://test.com"


class TestTrussService(TrussService):
    def __init__(self, remote_url: str, is_draft: bool, **kwargs):
        super().__init__(remote_url, is_draft, **kwargs)
        self._remote_url = remote_url

    def authenticate(self):
        return {"Authorization": "Test"}

    def is_live(self):
        response = self._send_request(self._remote_url, "GET")
        if response.status_code == 200:
            return True
        return False

    def is_ready(self):
        response = self._send_request(self._remote_url, "GET")
        if response.status_code == 200:
            return True
        return False


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
    service = TestTrussService(TEST_SERVICE_URL, True)
    assert service.is_live()


@mock.patch("requests.request", return_value=mock_unsuccessful_response())
def test_is_not_live(mock_request):
    service = TestTrussService(TEST_SERVICE_URL, True)
    assert service.is_live() is False


@mock.patch("requests.request", return_value=mock_successful_response())
def test_is_ready(mock_request):
    service = TestTrussService(TEST_SERVICE_URL, True)
    assert service.is_ready()


@mock.patch("requests.request", return_value=mock_unsuccessful_response())
def test_is_not_ready(mock_request):
    service = TestTrussService(TEST_SERVICE_URL, True)
    assert service.is_ready() is False


@mock.patch("requests.request", return_value=mock_successful_response())
def test_predict(mock_request):
    service = TestTrussService(TEST_SERVICE_URL, True)
    response = service.predict({"model_input": "test"})
    assert response.status_code == 200


@mock.patch("requests.request", return_value=mock_successful_response())
def test_predict_no_data(mock_request):
    service = TestTrussService(TEST_SERVICE_URL, True)
    with pytest.raises(ValueError):
        service._send_request(TEST_SERVICE_URL, "POST")
