from unittest import mock

import pytest
import requests
from requests import Response
from truss.remote.baseten.api import BasetenApi
from truss.remote.baseten.error import ApiError


@pytest.fixture
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


def mock_create_model_version_response():
    response = Response()
    response.status_code = 200
    response.json = mock.Mock(
        return_value={"data": {"create_model_version_from_truss": {"id": "12345"}}}
    )
    return response


def mock_create_model_response():
    response = Response()
    response.status_code = 200
    response.json = mock.Mock(
        return_value={"data": {"create_model_from_truss": {"id": "12345"}}}
    )
    return response


@pytest.fixture
def baseten_api(mock_auth_service):
    return BasetenApi("https://app.test.com", mock_auth_service)


@mock.patch("requests.post", return_value=mock_successful_response())
def test_post_graphql_query_success(mock_post, baseten_api):
    response_data = {"data": {"status": "success"}}

    result = baseten_api._post_graphql_query("sample_query_string")

    assert result == response_data


@mock.patch("requests.post", return_value=mock_graphql_error_response())
def test_post_graphql_query_error(mock_post, baseten_api):
    with pytest.raises(ApiError):
        baseten_api._post_graphql_query("sample_query_string")


@mock.patch("requests.post", return_value=mock_unsuccessful_response())
def test_post_requests_error(mock_post, baseten_api):
    with pytest.raises(requests.exceptions.HTTPError):
        baseten_api._post_graphql_query("sample_query_string")


@mock.patch("requests.post", return_value=mock_create_model_version_response())
def test_create_model_version_from_truss(mock_post, baseten_api):
    baseten_api.create_model_version_from_truss(
        "model_id",
        "s3key",
        "config_str",
        "semver_bump",
        "client_version",
        False,
        False,
        False,
        "deployment_name",
    )

    gql_mutation = mock_post.call_args[1]["data"]["query"]
    assert 'model_id: "model_id"' in gql_mutation
    assert 's3_key: "s3key"' in gql_mutation
    assert 'config: "config_str"' in gql_mutation
    assert 'semver_bump: "semver_bump"' in gql_mutation
    assert 'client_version: "client_version"' in gql_mutation
    assert "is_trusted: false" in gql_mutation
    assert "promote_after_deploy: false" in gql_mutation
    assert "scale_down_old_production: true" in gql_mutation
    assert 'name: "deployment_name"' in gql_mutation


@mock.patch("requests.post", return_value=mock_create_model_version_response())
def test_create_model_version_from_truss_does_not_send_deployment_name_if_not_specified(
    mock_post, baseten_api
):
    baseten_api.create_model_version_from_truss(
        "model_id",
        "s3key",
        "config_str",
        "semver_bump",
        "client_version",
        True,
        True,
        False,
        deployment_name=None,
    )

    gql_mutation = mock_post.call_args[1]["data"]["query"]
    assert 'model_id: "model_id"' in gql_mutation
    assert 's3_key: "s3key"' in gql_mutation
    assert 'config: "config_str"' in gql_mutation
    assert 'semver_bump: "semver_bump"' in gql_mutation
    assert 'client_version: "client_version"' in gql_mutation
    assert "is_trusted: true" in gql_mutation
    assert "promote_after_deploy: true" in gql_mutation
    assert "scale_down_old_production: true" in gql_mutation
    assert "name: " not in gql_mutation


@mock.patch("requests.post", return_value=mock_create_model_version_response())
def test_create_model_version_from_truss_does_not_scale_old_prod_to_zero_if_keep_previous_prod_settings(
    mock_post, baseten_api
):
    baseten_api.create_model_version_from_truss(
        "model_id",
        "s3key",
        "config_str",
        "semver_bump",
        "client_version",
        True,
        True,
        True,
        deployment_name=None,
    )

    gql_mutation = mock_post.call_args[1]["data"]["query"]
    assert 'model_id: "model_id"' in gql_mutation
    assert 's3_key: "s3key"' in gql_mutation
    assert 'config: "config_str"' in gql_mutation
    assert 'semver_bump: "semver_bump"' in gql_mutation
    assert 'client_version: "client_version"' in gql_mutation
    assert "is_trusted: true" in gql_mutation
    assert "promote_after_deploy: true" in gql_mutation
    assert "scale_down_old_production: false" in gql_mutation
    assert "name: " not in gql_mutation


@mock.patch("requests.post", return_value=mock_create_model_response())
def test_create_model_from_truss(mock_post, baseten_api):
    baseten_api.create_model_from_truss(
        "model_name",
        "s3key",
        "config_str",
        "semver_bump",
        "client_version",
        False,
        "deployment_name",
    )

    gql_mutation = mock_post.call_args[1]["data"]["query"]
    assert 'name: "model_name"' in gql_mutation
    assert 's3_key: "s3key"' in gql_mutation
    assert 'config: "config_str"' in gql_mutation
    assert 'semver_bump: "semver_bump"' in gql_mutation
    assert 'client_version: "client_version"' in gql_mutation
    assert "is_trusted: false" in gql_mutation
    assert 'version_name: "deployment_name"' in gql_mutation


@mock.patch("requests.post", return_value=mock_create_model_response())
def test_create_model_from_truss_does_not_send_deployment_name_if_not_specified(
    mock_post, baseten_api
):
    baseten_api.create_model_from_truss(
        "model_name",
        "s3key",
        "config_str",
        "semver_bump",
        "client_version",
        True,
        deployment_name=None,
    )

    gql_mutation = mock_post.call_args[1]["data"]["query"]
    assert 'name: "model_name"' in gql_mutation
    assert 's3_key: "s3key"' in gql_mutation
    assert 'config: "config_str"' in gql_mutation
    assert 'semver_bump: "semver_bump"' in gql_mutation
    assert 'client_version: "client_version"' in gql_mutation
    assert "is_trusted: true" in gql_mutation
    assert "version_name: " not in gql_mutation
