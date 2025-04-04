import pathlib
from unittest import mock

import pytest
import requests
from requests import Response

import truss_train.definitions as train_definitions
from truss.remote.baseten import custom_types as b10_types
from truss.remote.baseten.api import BasetenApi
from truss.remote.baseten.custom_types import ChainletDataAtomic, OracleData
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
        return_value={
            "data": {
                "create_model_version_from_truss": {
                    "model_version": {
                        "id": "12345",
                        "oracle": {
                            "id": "67890",
                            "name": "model-1",
                            "hostname": "localhost:1234",
                        },
                        "instance_type": {"name": "1x4"},
                    }
                }
            }
        }
    )
    return response


def mock_create_model_response():
    response = Response()
    response.status_code = 200
    response.json = mock.Mock(
        return_value={
            "data": {
                "create_model_from_truss": {
                    "model_version": {
                        "id": "12345",
                        "oracle": {
                            "id": "67890",
                            "name": "model-1",
                            "hostname": "localhost:1234",
                        },
                        "instance_type": {"name": "1x4"},
                    }
                }
            }
        }
    )
    return response


def mock_upsert_training_project_response():
    response = Response()
    response.status_code = 200
    response.json = mock.Mock(
        return_value={"training_project": {"id": "12345", "name": "training-project"}}
    )
    return response


def mock_create_development_model_response():
    response = Response()
    response.status_code = 200
    response.json = mock.Mock(
        return_value={
            "data": {"deploy_draft_truss": {"model_version": {"id": "12345"}}}
        }
    )
    return response


def mock_deploy_chain_deployment_response():
    response = Response()
    response.status_code = 200
    response.json = mock.Mock(
        return_value={
            "data": {
                "deploy_chain_atomic": {
                    "chain_deployment": {"id": "54321", "chain": {"id": "12345"}}
                }
            }
        }
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
        b10_types.TrussUserEnv.collect(),
        False,
        "deployment_name",
        "production",
    )

    gql_mutation = mock_post.call_args[1]["json"]["query"]
    assert 'model_id: "model_id"' in gql_mutation
    assert 's3_key: "s3key"' in gql_mutation
    assert 'config: "config_str"' in gql_mutation
    assert 'semver_bump: "semver_bump"' in gql_mutation
    assert {
        "trussUserEnv": b10_types.TrussUserEnv.collect().model_dump_json()
    } == mock_post.call_args[1]["json"]["variables"]
    assert "scale_down_old_production: true" in gql_mutation
    assert 'name: "deployment_name"' in gql_mutation
    assert 'environment_name: "production"' in gql_mutation
    assert "preserve_env_instance_type: false" in gql_mutation


@mock.patch("requests.post", return_value=mock_create_model_version_response())
def test_create_model_version_from_truss_does_not_send_deployment_name_if_not_specified(
    mock_post, baseten_api
):
    baseten_api.create_model_version_from_truss(
        "model_id",
        "s3key",
        "config_str",
        "semver_bump",
        b10_types.TrussUserEnv.collect(),
        False,
        deployment_name=None,
        preserve_env_instance_type=False,
    )

    gql_mutation = mock_post.call_args[1]["json"]["query"]
    assert 'model_id: "model_id"' in gql_mutation
    assert 's3_key: "s3key"' in gql_mutation
    assert 'config: "config_str"' in gql_mutation
    assert 'semver_bump: "semver_bump"' in gql_mutation
    assert {
        "trussUserEnv": b10_types.TrussUserEnv.collect().model_dump_json()
    } == mock_post.call_args[1]["json"]["variables"]
    assert "scale_down_old_production: true" in gql_mutation
    assert " name: " not in gql_mutation
    assert "environment_name: " not in gql_mutation
    assert "preserve_env_instance_type: false" in gql_mutation


@mock.patch("requests.post", return_value=mock_create_model_version_response())
def test_create_model_version_from_truss_does_not_scale_old_prod_to_zero_if_keep_previous_prod_settings(
    mock_post, baseten_api
):
    baseten_api.create_model_version_from_truss(
        "model_id",
        "s3key",
        "config_str",
        "semver_bump",
        b10_types.TrussUserEnv.collect(),
        True,
        deployment_name=None,
        environment="staging",
        preserve_env_instance_type=True,
    )

    gql_mutation = mock_post.call_args[1]["json"]["query"]

    assert 'model_id: "model_id"' in gql_mutation
    assert 's3_key: "s3key"' in gql_mutation
    assert 'config: "config_str"' in gql_mutation
    assert 'semver_bump: "semver_bump"' in gql_mutation
    assert {
        "trussUserEnv": b10_types.TrussUserEnv.collect().model_dump_json()
    } == mock_post.call_args[1]["json"]["variables"]
    assert "scale_down_old_production: false" in gql_mutation
    assert " name: " not in gql_mutation
    assert 'environment_name: "staging"' in gql_mutation
    assert "preserve_env_instance_type: true" in gql_mutation


@mock.patch("requests.post", return_value=mock_create_model_response())
def test_create_model_from_truss(mock_post, baseten_api):
    baseten_api.create_model_from_truss(
        "model_name",
        "s3key",
        "config_str",
        "semver_bump",
        b10_types.TrussUserEnv.collect(),
        deployment_name="deployment_name",
    )

    gql_mutation = mock_post.call_args[1]["json"]["query"]
    assert 'name: "model_name"' in gql_mutation
    assert 's3_key: "s3key"' in gql_mutation
    assert 'config: "config_str"' in gql_mutation
    assert 'semver_bump: "semver_bump"' in gql_mutation
    assert {
        "trussUserEnv": b10_types.TrussUserEnv.collect().model_dump_json()
    } == mock_post.call_args[1]["json"]["variables"]
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
        b10_types.TrussUserEnv.collect(),
        deployment_name=None,
    )

    gql_mutation = mock_post.call_args[1]["json"]["query"]
    assert 'name: "model_name"' in gql_mutation
    assert 's3_key: "s3key"' in gql_mutation
    assert 'config: "config_str"' in gql_mutation
    assert 'semver_bump: "semver_bump"' in gql_mutation
    assert {
        "trussUserEnv": b10_types.TrussUserEnv.collect().model_dump_json()
    } == mock_post.call_args[1]["json"]["variables"]
    assert "version_name: " not in gql_mutation


@mock.patch("requests.post", return_value=mock_create_model_response())
def test_create_model_from_truss_with_allow_truss_download(mock_post, baseten_api):
    baseten_api.create_model_from_truss(
        "model_name",
        "s3key",
        "config_str",
        "semver_bump",
        b10_types.TrussUserEnv.collect(),
        allow_truss_download=False,
    )

    gql_mutation = mock_post.call_args[1]["json"]["query"]
    assert 'name: "model_name"' in gql_mutation
    assert 's3_key: "s3key"' in gql_mutation
    assert 'config: "config_str"' in gql_mutation
    assert 'semver_bump: "semver_bump"' in gql_mutation
    assert {
        "trussUserEnv": b10_types.TrussUserEnv.collect().model_dump_json()
    } == mock_post.call_args[1]["json"]["variables"]
    assert "allow_truss_download: false" in gql_mutation


@mock.patch("requests.post", return_value=mock_create_development_model_response())
def test_create_development_model_from_truss_with_allow_truss_download(
    mock_post, baseten_api
):
    baseten_api.create_development_model_from_truss(
        "model_name",
        "s3key",
        "config_str",
        b10_types.TrussUserEnv.collect(),
        allow_truss_download=False,
    )

    gql_mutation = mock_post.call_args[1]["json"]["query"]
    assert 'name: "model_name"' in gql_mutation
    assert 's3_key: "s3key"' in gql_mutation
    assert 'config: "config_str"' in gql_mutation
    assert {
        "trussUserEnv": b10_types.TrussUserEnv.collect().model_dump_json()
    } == mock_post.call_args[1]["json"]["variables"]
    assert "allow_truss_download: false" in gql_mutation


@mock.patch("requests.post", return_value=mock_deploy_chain_deployment_response())
def test_deploy_chain_deployment(mock_post, baseten_api):
    baseten_api.deploy_chain_atomic(
        environment="production",
        chain_id="chain_id",
        dependencies=[],
        entrypoint=ChainletDataAtomic(
            name="chainlet-1",
            oracle=OracleData(
                model_name="model-1",
                s3_key="s3-key-1",
                encoded_config_str="encoded-config-str-1",
            ),
        ),
        truss_user_env=b10_types.TrussUserEnv.collect(),
    )

    gql_mutation = mock_post.call_args[1]["json"]["query"]

    assert 'environment: "production"' in gql_mutation
    assert 'chain_id: "chain_id"' in gql_mutation
    assert "dependencies:" in gql_mutation
    assert "entrypoint:" in gql_mutation


@mock.patch("requests.post", return_value=mock_deploy_chain_deployment_response())
def test_deploy_chain_deployment_with_gitinfo(mock_post, baseten_api):
    baseten_api.deploy_chain_atomic(
        environment="production",
        chain_id="chain_id",
        dependencies=[],
        entrypoint=ChainletDataAtomic(
            name="chainlet-1",
            oracle=OracleData(
                model_name="model-1",
                s3_key="s3-key-1",
                encoded_config_str="encoded-config-str-1",
            ),
        ),
        truss_user_env=b10_types.TrussUserEnv.collect_with_git_info(pathlib.Path(".")),
    )

    gql_mutation = mock_post.call_args[1]["json"]["query"]

    assert 'environment: "production"' in gql_mutation
    assert 'chain_id: "chain_id"' in gql_mutation
    assert "dependencies:" in gql_mutation
    assert "entrypoint:" in gql_mutation


@mock.patch("requests.post", return_value=mock_deploy_chain_deployment_response())
def test_deploy_chain_deployment_no_environment(mock_post, baseten_api):
    baseten_api.deploy_chain_atomic(
        chain_id="chain_id",
        dependencies=[],
        entrypoint=ChainletDataAtomic(
            name="chainlet-1",
            oracle=OracleData(
                model_name="model-1",
                s3_key="s3-key-1",
                encoded_config_str="encoded-config-str-1",
            ),
        ),
        truss_user_env=b10_types.TrussUserEnv.collect(),
    )

    gql_mutation = mock_post.call_args[1]["json"]["query"]

    assert 'chain_id: "chain_id"' in gql_mutation
    assert "environment" not in gql_mutation
    assert "dependencies:" in gql_mutation
    assert "entrypoint:" in gql_mutation


@mock.patch("requests.post", return_value=mock_upsert_training_project_response())
def test_upsert_training_project(mock_post, baseten_api):
    baseten_api.upsert_training_project(
        training_project=train_definitions.TrainingProject(
            name="training-project",
            job=train_definitions.TrainingJob(
                image=train_definitions.Image(base_image="base-image"),
                runtime_config=train_definitions.Runtime(
                    start_commands=["/bin/bash entrypoint.sh"]
                ),
            ),
        )
    )

    upsert_body = mock_post.call_args[1]["json"]["training_project"]
    assert "job" not in upsert_body
    assert "training-project" == upsert_body["name"]
