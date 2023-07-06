from unittest import mock

import pytest
import requests
from requests import Response
from truss.remote.baseten.api import BasetenApi
from truss.remote.baseten.error import ApiError


def mock_auth_service():
    auth_service = mock.Mock()
    auth_token = mock.Mock(headers=lambda: {"Authorization": "Api-Key token"})
    auth_service.authenticate.return_value = auth_token
    return auth_service


def mock_successful_response():
    response = Response()
    response.status_code = 200
    response.json = mock.Mock(return_value={"data": {"status": "success"}})
    return response


def mock_graphql_error_response():
    response = Response()
    response.status_code = 200
    response.json = mock.Mock(return_value={"errors": [{"message": "error"}]})
    return response


def mock_unsuccessful_response():
    response = Response()
    response.status_code = 400
    return response


@mock.patch("truss.remote.baseten.auth.AuthService")
@mock.patch("requests.post", return_value=mock_successful_response())
def test_post_graphql_query_success(mock_post, mock_auth_service):
    api_url = "https://test.com/api"
    api = BasetenApi(api_url, mock_auth_service)

    response_data = {"data": {"status": "success"}}

    result = api._post_graphql_query("sample_query_string")

    assert result == response_data


@mock.patch("truss.remote.baseten.auth.AuthService")
@mock.patch("requests.post", return_value=mock_graphql_error_response())
def test_post_graphql_query_error(mock_post, mock_auth_service):
    api_url = "https://test.com/api"
    api = BasetenApi(api_url, mock_auth_service)

    with pytest.raises(ApiError):
        api._post_graphql_query("sample_query_string")


@mock.patch("truss.remote.baseten.auth.AuthService")
@mock.patch("requests.post", return_value=mock_unsuccessful_response())
def test_post_requests_error(mock_post, mock_auth_service):
    api_url = "https://test.com/api"
    api = BasetenApi(api_url, mock_auth_service)
    with pytest.raises(requests.exceptions.HTTPError):
        api._post_graphql_query("sample_query_string")
