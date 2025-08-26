import pathlib
from unittest import mock

import pytest
import requests
from requests import Response

import truss_train.definitions as train_definitions
from truss.remote.baseten import custom_types as b10_types
from truss.remote.baseten.api import BasetenApi
from truss.remote.baseten.core import get_training_job_logs_with_pagination
from truss.remote.baseten.custom_types import ChainletDataAtomic, OracleData
from truss.remote.baseten.error import ApiError
from truss.remote.baseten.utils.time import iso_to_millis


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


def mock_graphql_error_response_with_description():
    response = Response()
    response.status_code = 200
    response.json = mock.Mock(
        return_value={
            "errors": [
                {"message": "error", "extensions": {"description": "descriptive_error"}}
            ]
        }
    )
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


@mock.patch(
    "requests.post", return_value=mock_graphql_error_response_with_description()
)
def test_post_graphql_query_error_with_description(mock_post, baseten_api):
    with pytest.raises(ApiError) as exc_info:
        baseten_api._post_graphql_query("sample_query_string")

    exception = exc_info.value
    assert str(exception) == "descriptive_error"


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
    assert "preserve_env_instance_type: true" in gql_mutation


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
                runtime=train_definitions.Runtime(
                    start_commands=["/bin/bash entrypoint.sh"]
                ),
            ),
        )
    )

    upsert_body = mock_post.call_args[1]["json"]["training_project"]
    assert "job" not in upsert_body
    assert "training-project" == upsert_body["name"]


# Mock responses for training job logs pagination tests
def mock_training_job_logs_response(logs, has_more=True):
    """Helper function to create mock training job logs response"""
    response = Response()
    response.status_code = 200
    response.json = mock.Mock(return_value={"logs": logs})
    return response


def mock_training_job_logs_empty_response():
    """Helper function to create mock empty training job logs response"""
    response = Response()
    response.status_code = 200
    response.json = mock.Mock(return_value={"logs": []})
    return response


def mock_training_job_logs_error_response():
    """Helper function to create mock error response for training job logs"""
    response = Response()
    response.status_code = 500
    response.raise_for_status = mock.Mock(
        side_effect=requests.exceptions.HTTPError("Server Error")
    )
    return response


def test_get_training_job_logs_with_pagination_single_batch(baseten_api):
    """Test pagination when all logs fit in a single batch"""
    # Mock logs data
    now_as_iso = "2022-01-01T00:00:00Z"
    now_as_millis = iso_to_millis(now_as_iso)
    mock_logs = [
        {"timestamp": now_as_millis, "message": "Log 1"},
        {"timestamp": now_as_millis + 60000, "message": "Log 2"},
        {"timestamp": now_as_millis + 120000, "message": "Log 3"},
    ]

    # Mock the _fetch_log_batch method to return logs on first call, empty on second
    mock_fetch = mock.Mock(side_effect=[mock_logs, []])
    baseten_api._fetch_log_batch = mock_fetch

    # Mock get_training_job method
    baseten_api.get_training_job = mock.Mock(
        return_value={"training_job": {"created_at": "2022-01-01T00:00:00Z"}}
    )

    result = get_training_job_logs_with_pagination(
        baseten_api, "project-123", "job-456", batch_size=5
    )

    # Verify the result
    assert result == mock_logs
    assert len(result) == 3

    # Verify the mock was called twice (once for logs, once for empty batch)
    assert mock_fetch.call_count == 2

    # Verify the first call parameters
    first_call_args = mock_fetch.call_args_list[0]
    assert first_call_args[0][0] == "project-123"  # project_id
    assert first_call_args[0][1] == "job-456"  # job_id

    # Verify the query body contains expected parameters
    query_params = first_call_args[0][2]  # query_params
    assert query_params["limit"] == 5  # batch_size
    assert query_params["direction"] == "asc"
    assert "start_epoch_millis" in query_params
    assert "end_epoch_millis" in query_params


def test_get_training_job_logs_with_pagination_multiple_batches(baseten_api):
    """Test pagination when logs span multiple batches"""
    # Mock logs data for multiple batches
    batch1_logs = [
        {"timestamp": "1640995200000000000", "message": "Log 1"},  # 2022-01-01 00:00:00
        {"timestamp": "1640995260000000000", "message": "Log 2"},  # 2022-01-01 00:01:00
    ]
    batch2_logs = [
        {"timestamp": "1640995320000000000", "message": "Log 3"},  # 2022-01-01 00:02:00
        {"timestamp": "1640995380000000000", "message": "Log 4"},  # 2022-01-01 00:03:00
    ]
    batch3_logs = [
        {"timestamp": "1640995440000000000", "message": "Log 5"}  # 2022-01-01 00:04:00
    ]

    # Mock the rest_api_client directly on the instance
    mock_rest_client = mock.Mock()
    mock_rest_client.post.side_effect = [
        {"logs": batch1_logs},
        {"logs": batch2_logs},
        {"logs": batch3_logs},
    ]
    baseten_api._rest_api_client = mock_rest_client

    # Mock get_training_job method
    baseten_api.get_training_job = mock.Mock(
        return_value={"training_job": {"created_at": "2022-01-01T00:00:00Z"}}
    )

    result = get_training_job_logs_with_pagination(
        baseten_api, "project-123", "job-456", batch_size=2
    )

    # Verify the result
    expected_logs = batch1_logs + batch2_logs + batch3_logs
    assert result == expected_logs
    assert len(result) == 5

    # Verify the API calls
    assert mock_rest_client.post.call_count == 3

    # Verify first call
    first_call = mock_rest_client.post.call_args_list[0]
    assert first_call[0][0] == "v1/training_projects/project-123/jobs/job-456/logs"
    first_query_body = first_call[1]["body"]
    assert first_query_body["limit"] == 2
    assert first_query_body["direction"] == "asc"

    # Verify second call (should use timestamp from last log of first batch)
    second_call = mock_rest_client.post.call_args_list[1]
    second_query_body = second_call[1]["body"]
    assert second_query_body["start_epoch_millis"] == 1640995260001  # timestamp + 1ms
    assert second_query_body["limit"] == 2
    assert second_query_body["direction"] == "asc"

    # Verify third call
    third_call = mock_rest_client.post.call_args_list[2]
    third_query_body = third_call[1]["body"]
    assert third_query_body["start_epoch_millis"] == 1640995380001  # timestamp + 1ms


def test_get_training_job_logs_with_pagination_empty_response(baseten_api):
    """Test pagination when no logs are returned"""
    # Mock the rest_api_client directly on the instance
    mock_rest_client = mock.Mock()
    mock_rest_client.post.return_value = {"logs": []}
    baseten_api._rest_api_client = mock_rest_client

    # Mock get_training_job method
    baseten_api.get_training_job = mock.Mock(
        return_value={"training_job": {"created_at": "2022-01-01T00:00:00Z"}}
    )

    result = get_training_job_logs_with_pagination(
        baseten_api, "project-123", "job-456"
    )

    # Verify the result
    assert result == []
    assert len(result) == 0

    # Verify the API call
    mock_rest_client.post.assert_called_once()


def test_get_training_job_logs_with_pagination_partial_batch(baseten_api):
    """Test pagination when the last batch has fewer logs than batch_size"""
    batch1_logs = [
        {"timestamp": "1640995200000000000", "message": "Log 1"},
        {"timestamp": "1640995260000000000", "message": "Log 2"},
    ]
    batch2_logs = [{"timestamp": "1640995320000000000", "message": "Log 3"}]

    # Mock the rest_api_client directly on the instance
    mock_rest_client = mock.Mock()
    mock_rest_client.post.side_effect = [{"logs": batch1_logs}, {"logs": batch2_logs}]
    baseten_api._rest_api_client = mock_rest_client

    # Mock get_training_job method
    baseten_api.get_training_job = mock.Mock(
        return_value={"training_job": {"created_at": "2022-01-01T00:00:00Z"}}
    )

    result = get_training_job_logs_with_pagination(
        baseten_api, "project-123", "job-456", batch_size=2
    )

    # Verify the result
    expected_logs = batch1_logs + batch2_logs
    assert result == expected_logs
    assert len(result) == 3

    # Verify only 2 API calls (should stop after partial batch)
    assert mock_rest_client.post.call_count == 2


def test_get_training_job_logs_with_pagination_max_iterations(baseten_api):
    """Test pagination when maximum iterations are reached"""
    # Mock logs that would cause infinite pagination
    batch_logs = [
        {"timestamp": "1640995200000000000", "message": "Log 1"},
        {"timestamp": "1640995260000000000", "message": "Log 2"},
    ]

    # Mock the rest_api_client directly on the instance
    mock_rest_client = mock.Mock()
    # Configure mock to always return the same batch (simulating infinite pagination)
    mock_rest_client.post.return_value = {"logs": batch_logs}
    baseten_api._rest_api_client = mock_rest_client

    # Mock get_training_job method
    baseten_api.get_training_job = mock.Mock(
        return_value={"training_job": {"created_at": "2022-01-01T00:00:00Z"}}
    )

    result = get_training_job_logs_with_pagination(
        baseten_api, "project-123", "job-456", batch_size=2
    )

    # Verify the result (should have MAX_ITERATIONS * batch_size logs)
    from truss.remote.baseten.core import MAX_ITERATIONS

    expected_log_count = MAX_ITERATIONS * 2
    assert len(result) == expected_log_count

    # Verify MAX_ITERATIONS API calls were made
    assert mock_rest_client.post.call_count == MAX_ITERATIONS


def test_get_training_job_logs_with_pagination_api_error(baseten_api):
    """Test pagination when API returns an error"""
    # Mock the rest_api_client directly on the instance
    mock_rest_client = mock.Mock()
    # Configure mock to raise an exception
    mock_rest_client.post.side_effect = Exception("API Error")
    baseten_api._rest_api_client = mock_rest_client

    # Mock get_training_job method
    baseten_api.get_training_job = mock.Mock(
        return_value={"training_job": {"created_at": "2022-01-01T00:00:00Z"}}
    )

    result = get_training_job_logs_with_pagination(
        baseten_api, "project-123", "job-456"
    )

    # Verify the result is empty when error occurs
    assert result == []
    assert len(result) == 0

    # Verify the API call was attempted
    mock_rest_client.post.assert_called_once()


def test_get_training_job_logs_with_pagination_custom_batch_size(baseten_api):
    """Test pagination with custom batch size"""
    mock_logs = [
        {"timestamp": "1640995200000000000", "message": "Log 1"},
        {"timestamp": "1640995260000000000", "message": "Log 2"},
    ]

    # Mock the rest_api_client directly on the instance
    mock_rest_client = mock.Mock()
    mock_rest_client.post.return_value = {"logs": mock_logs}
    baseten_api._rest_api_client = mock_rest_client

    # Mock get_training_job method
    baseten_api.get_training_job = mock.Mock(
        return_value={"training_job": {"created_at": "2022-01-01T00:00:00Z"}}
    )

    result = get_training_job_logs_with_pagination(
        baseten_api, "project-123", "job-456", batch_size=50
    )

    # Verify the result
    assert result == mock_logs

    # Verify the API call used custom batch size
    call_args = mock_rest_client.post.call_args
    query_body = call_args[1]["body"]
    assert query_body["limit"] == 50


def test_get_training_job_logs_with_pagination_six_batches(baseten_api):
    """Test pagination with six batches"""
    iso_time = "2022-01-01T00:00:00Z"
    now_as_millis = iso_to_millis(iso_time)
    mock_logs_batch_1 = [
        {"timestamp": now_as_millis + 1000, "message": "Log 1"},
        {"timestamp": now_as_millis + 2000, "message": "Log 2"},
    ]
    mock_logs_batch_2 = [
        {"timestamp": now_as_millis + 3000, "message": "Log 3"},
        {"timestamp": now_as_millis + 4000, "message": "Log 4"},
    ]
    mock_logs_batch_3 = [{"timestamp": now_as_millis + 5000, "message": "Log 5"}]

    # Mock the rest_api_client directly on the instance
    mock_rest_client = mock.Mock()
    mock_rest_client.post.side_effect = [
        {"logs": mock_logs_batch_1},
        {"logs": mock_logs_batch_2},
        {"logs": mock_logs_batch_3},
    ]
    baseten_api._rest_api_client = mock_rest_client

    # Mock get_training_job method
    baseten_api.get_training_job = mock.Mock(
        return_value={"training_job": {"created_at": "2022-01-01T00:00:00Z"}}
    )

    result = get_training_job_logs_with_pagination(
        baseten_api, "project-123", "job-456", batch_size=2
    )

    # Verify the result
    assert result == mock_logs_batch_1 + mock_logs_batch_2 + mock_logs_batch_3
    assert mock_rest_client.post.call_count == 3 + 1


def test_get_training_job_logs_with_pagination_timestamp_conversion(baseten_api):
    """Test that timestamp conversion from nanoseconds to milliseconds works correctly"""
    batch1_logs = [
        {"timestamp": "1640995200000000000", "message": "Log 1"}  # 1640995200000 ms
    ]
    batch2_logs = [
        {"timestamp": "1640995260000000000", "message": "Log 2"}  # 1640995260000 ms
    ]

    # Mock the rest_api_client directly on the instance
    mock_rest_client = mock.Mock()
    mock_rest_client.post.side_effect = [{"logs": batch1_logs}, {"logs": batch2_logs}]
    baseten_api._rest_api_client = mock_rest_client

    # Mock get_training_job method
    baseten_api.get_training_job = mock.Mock(
        return_value={"training_job": {"created_at": "2022-01-01T00:00:00Z"}}
    )

    result = get_training_job_logs_with_pagination(
        baseten_api, "project-123", "job-456", batch_size=1
    )

    # Verify the result
    expected_logs = batch1_logs + batch2_logs
    assert result == expected_logs

    # Verify the second call uses correct timestamp conversion
    second_call = mock_rest_client.post.call_args_list[1]
    second_query_body = second_call[1]["body"]
    # Should be 1640995200000 + 1 = 1640995200001
    assert second_query_body["start_epoch_millis"] == 1640995200001


def test_get_training_job_logs_with_pagination_query_body_filtering(baseten_api):
    """Test that None values are properly filtered from query body"""
    mock_logs = [{"timestamp": "1640995200000000000", "message": "Log 1"}]

    # Mock the rest_api_client directly on the instance
    mock_rest_client = mock.Mock()
    mock_rest_client.post.return_value = {"logs": mock_logs}
    baseten_api._rest_api_client = mock_rest_client

    # Mock get_training_job method
    baseten_api.get_training_job = mock.Mock(
        return_value={"training_job": {"created_at": "2022-01-01T00:00:00Z"}}
    )

    get_training_job_logs_with_pagination(baseten_api, "project-123", "job-456")

    # Verify the API call
    call_args = mock_rest_client.post.call_args
    query_body = call_args[1]["body"]

    # Verify that all required values are included in the query body
    assert "start_epoch_millis" in query_body
    assert "end_epoch_millis" in query_body
    assert "limit" in query_body
    assert "direction" in query_body


# Tests for new helper methods
def test_build_log_query_params(baseten_api):
    """Test _build_log_query_params helper method"""
    from truss.remote.baseten.core import _build_log_query_params

    # Test with all parameters
    query_params = _build_log_query_params(
        start_time=1640995200000, end_time=1640995260000, batch_size=100
    )

    expected = {
        "start_epoch_millis": 1640995200000,
        "end_epoch_millis": 1640995260000,
        "limit": 100,
        "direction": "asc",
    }
    assert query_params == expected

    # Test with None values (should be filtered out)
    query_params = _build_log_query_params(
        start_time=None, end_time=None, batch_size=50
    )

    expected = {"limit": 50, "direction": "asc"}
    assert query_params == expected


def test_fetch_log_batch(baseten_api):
    """Test _fetch_log_batch helper method"""

    mock_logs = [
        {"timestamp": "1640995200000000000", "message": "Log 1"},
        {"timestamp": "1640995260000000000", "message": "Log 2"},
    ]

    # Mock the rest_api_client
    mock_rest_client = mock.Mock()
    mock_rest_client.post.return_value = {"logs": mock_logs}
    baseten_api._rest_api_client = mock_rest_client

    query_params = {"limit": 100, "direction": "asc"}
    result = baseten_api._fetch_log_batch("project-123", "job-456", query_params)

    assert result == mock_logs
    mock_rest_client.post.assert_called_with(
        "v1/training_projects/project-123/jobs/job-456/logs", body=query_params
    )


def test_handle_server_error_backoff(baseten_api):
    """Test _handle_server_error_backoff helper method"""
    from truss.remote.baseten.core import _handle_server_error_backoff

    # Create a mock HTTP error
    mock_response = mock.Mock()
    mock_response.status_code = 500

    mock_error = requests.HTTPError("Server Error")
    mock_error.response = mock_response

    # Test backoff behavior
    new_batch_size = _handle_server_error_backoff(mock_error, "job-456", 1, 1000)

    # Should reduce batch size by half
    assert new_batch_size == 500

    # Test minimum batch size
    new_batch_size = _handle_server_error_backoff(mock_error, "job-456", 2, 150)

    # Should not go below 100
    assert new_batch_size == 100


def test_process_batch_logs_continue(baseten_api):
    """Test _process_batch_logs when pagination should continue"""
    from truss.remote.baseten.core import _process_batch_logs

    batch_logs = [
        {"timestamp": "1640995200000000000", "message": "Log 1"},
        {"timestamp": "1640995260000000000", "message": "Log 2"},
    ]

    should_continue, next_start_time, next_end_time = _process_batch_logs(
        batch_logs, "job-456", 1, 2
    )

    assert should_continue is True
    # Should be 1640995260000 + 1 = 1640995260001 (last timestamp + 1ms)
    assert next_start_time == 1640995260001


def test_process_batch_logs_empty(baseten_api):
    """Test _process_batch_logs when batch is empty"""
    from truss.remote.baseten.core import _process_batch_logs

    should_continue, next_start_time, next_end_time = _process_batch_logs(
        [], "job-456", 1, 100
    )

    assert should_continue is False
    assert next_start_time is None


def test_process_batch_logs_partial(baseten_api):
    """Test _process_batch_logs when batch is smaller than expected"""
    from truss.remote.baseten.core import _process_batch_logs

    batch_logs = [{"timestamp": "1640995200000000000", "message": "Log 1"}]

    should_continue, next_start_time, next_end_time = _process_batch_logs(
        batch_logs, "job-456", 1, 100
    )

    assert should_continue is True
    assert next_start_time is not None
    assert next_end_time is not None


def test_get_training_job_logs_with_pagination_server_error_retry(baseten_api):
    """Test pagination with server error retry logic"""
    batch_logs = [
        {"timestamp": "1640995200000000000", "message": "Log 1"},
        {"timestamp": "1640995260000000000", "message": "Log 2"},
    ]

    # Mock the rest_api_client
    mock_rest_client = mock.Mock()

    # First call fails with 500, second call succeeds
    mock_response_500 = mock.Mock()
    mock_response_500.status_code = 500
    mock_error_500 = requests.HTTPError("Server Error")
    mock_error_500.response = mock_response_500

    mock_rest_client.post.side_effect = [
        mock_error_500,  # First call fails
        {"logs": batch_logs},  # Second call succeeds
    ]
    baseten_api._rest_api_client = mock_rest_client

    # Mock get_training_job method
    baseten_api.get_training_job = mock.Mock(
        return_value={"training_job": {"created_at": "2022-01-01T00:00:00Z"}}
    )

    result = get_training_job_logs_with_pagination(
        baseten_api, "project-123", "job-456", batch_size=1000
    )

    # Should get the logs after retry
    assert result == batch_logs

    # Should have made 2 calls
    assert mock_rest_client.post.call_count == 3
    # Second call should use reduced batch size
    second_call = mock_rest_client.post.call_args_list[1]
    second_query_body = second_call[1]["body"]
    assert second_query_body["limit"] == 500  # Reduced from 1000


def test_get_training_job_logs_with_pagination_non_server_error(baseten_api):
    """Test pagination with non-server error (should not retry)"""
    # Mock the rest_api_client
    mock_rest_client = mock.Mock()

    # Mock a 400 error (client error, not server error)
    mock_response_400 = mock.Mock()
    mock_response_400.status_code = 400
    mock_error_400 = requests.HTTPError("Bad Request")
    mock_error_400.response = mock_response_400

    mock_rest_client.post.side_effect = mock_error_400
    baseten_api._rest_api_client = mock_rest_client

    # Mock get_training_job method
    baseten_api.get_training_job = mock.Mock(
        return_value={"training_job": {"created_at": "2022-01-01T00:00:00Z"}}
    )

    result = get_training_job_logs_with_pagination(
        baseten_api, "project-123", "job-456"
    )

    # Should return empty list on non-server error
    assert result == []

    # Should have made only 1 call (no retry)
    assert mock_rest_client.post.call_count == 1


def test_get_training_job_logs_with_pagination_default_batch_size(baseten_api):
    """Test that default batch size is MAX_BATCH_SIZE"""
    mock_logs = [{"timestamp": "1640995200000000000", "message": "Log 1"}]

    # Mock the rest_api_client
    mock_rest_client = mock.Mock()
    mock_rest_client.post.return_value = {"logs": mock_logs}
    baseten_api._rest_api_client = mock_rest_client

    # Mock get_training_job method
    baseten_api.get_training_job = mock.Mock(
        return_value={"training_job": {"created_at": "2022-01-01T00:00:00Z"}}
    )

    get_training_job_logs_with_pagination(baseten_api, "project-123", "job-456")

    # Verify the API call used default batch size
    call_args = mock_rest_client.post.call_args
    query_body = call_args[1]["body"]

    from truss.remote.baseten.core import MAX_BATCH_SIZE

    assert query_body["limit"] == MAX_BATCH_SIZE
