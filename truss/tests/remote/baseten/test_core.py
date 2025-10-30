import json
from tempfile import NamedTemporaryFile
from unittest import mock
from unittest.mock import MagicMock

import pytest
import requests

from truss.base.constants import PRODUCTION_ENVIRONMENT_NAME
from truss.base.errors import ValidationError
from truss.remote.baseten import core
from truss.remote.baseten import custom_types as b10_types
from truss.remote.baseten.api import BasetenApi
from truss.remote.baseten.core import (
    MAX_BATCH_SIZE,
    create_truss_service,
    get_training_job_logs_with_pagination,
)
from truss.remote.baseten.error import ApiError
from truss.remote.baseten.utils.time import iso_to_millis


def test_exists_model():
    def mock_get_model(model_name):
        if model_name == "first model":
            return {"model": {"id": "1"}}
        elif model_name == "second model":
            return {"model": {"id": "2"}}
        else:
            raise ApiError(
                "Oracle not found",
                BasetenApi.GraphQLErrorCodes.RESOURCE_NOT_FOUND.value,
            )

    api = MagicMock()
    api.get_model.side_effect = mock_get_model

    assert core.exists_model(api, "first model")
    assert core.exists_model(api, "second model")
    assert not core.exists_model(api, "third model")


def test_upload_truss():
    api = MagicMock()
    api.model_s3_upload_credentials.return_value = {
        "s3_key": "key",
        "s3_bucket": "bucket",
    }
    core.multipart_upload_boto3 = MagicMock()
    core.multipart_upload_boto3.return_value = None
    test_file = NamedTemporaryFile()
    assert core.upload_truss(api, test_file, None) == "key"


def test_get_dev_version_from_versions():
    versions = [{"id": "1", "is_draft": False}, {"id": "2", "is_draft": True}]
    dev_version = core.get_dev_version_from_versions(versions)
    assert dev_version["id"] == "2"


def test_get_dev_version_from_versions_error():
    versions = [{"id": "1", "is_draft": False}]
    dev_version = core.get_dev_version_from_versions(versions)
    assert dev_version is None


def test_get_dev_version():
    versions = [{"id": "1", "is_draft": False}, {"id": "2", "is_draft": True}]
    api = MagicMock()
    api.get_model.return_value = {"model": {"versions": versions}}

    dev_version = core.get_dev_version(api, "my_model")
    assert dev_version["id"] == "2"


def test_get_prod_version_from_versions():
    versions = [
        {"id": "1", "is_draft": False, "is_primary": False},
        {"id": "2", "is_draft": True, "is_primary": False},
        {"id": "3", "is_draft": False, "is_primary": True},
    ]
    prod_version = core.get_prod_version_from_versions(versions)
    assert prod_version["id"] == "3"


def test_get_prod_version_from_versions_error():
    versions = [
        {"id": "1", "is_draft": True, "is_primary": False},
        {"id": "2", "is_draft": False, "is_primary": False},
    ]
    prod_version = core.get_prod_version_from_versions(versions)
    assert prod_version is None


@pytest.mark.parametrize("environment", [None, PRODUCTION_ENVIRONMENT_NAME])
def test_create_truss_service_handles_eligible_environment_values(environment):
    api = MagicMock()
    return_value = {
        "id": "model_version_id",
        "oracle": {"id": "model_id", "hostname": "hostname"},
    }
    api.create_model_from_truss.return_value = return_value
    version_handle = create_truss_service(
        api,
        "model_name",
        "s3_key",
        "config",
        b10_types.TrussUserEnv.collect(),
        preserve_previous_prod_deployment=False,
        is_draft=False,
        model_id=None,
        deployment_name="deployment_name",
        environment=environment,
    )
    assert version_handle.version_id == "model_version_id"
    assert version_handle.model_id == "model_id"
    api.create_model_from_truss.assert_called_once()


@pytest.mark.parametrize("model_id", ["some_model_id", None])
def test_create_truss_services_handles_is_draft(model_id):
    api = MagicMock()
    return_value = {
        "id": "model_version_id",
        "oracle": {"id": "model_id", "hostname": "hostname"},
        "instance_type": {"name": "1x2"},
    }
    api.create_development_model_from_truss.return_value = return_value
    version_handle = create_truss_service(
        api,
        "model_name",
        "s3_key",
        "config",
        b10_types.TrussUserEnv.collect(),
        preserve_previous_prod_deployment=False,
        is_draft=True,
        model_id=model_id,
        deployment_name="deployment_name",
    )
    assert version_handle.version_id == "model_version_id"
    assert version_handle.model_id == "model_id"
    api.create_development_model_from_truss.assert_called_once()


@pytest.mark.parametrize(
    "inputs",
    [
        {
            "environment": None,
            "deployment_name": "some deployment",
            "preserve_previous_prod_deployment": False,
        },
        {
            "environment": PRODUCTION_ENVIRONMENT_NAME,
            "deployment_name": None,
            "preserve_previous_prod_deployment": False,
        },
        {
            "environment": "staging",
            "deployment_name": "some_deployment_name",
            "preserve_previous_prod_deployment": True,
        },
    ],
)
def test_create_truss_service_handles_existing_model(inputs):
    api = MagicMock()
    return_value = {
        "id": "model_version_id",
        "oracle": {"id": "model_id", "hostname": "hostname"},
        "instance_type": {"name": "1x2"},
    }
    api.create_model_version_from_truss.return_value = return_value
    version_handle = create_truss_service(
        api,
        "model_name",
        "s3_key",
        "config",
        b10_types.TrussUserEnv.collect(),
        is_draft=False,
        model_id="model_id",
        **inputs,
    )

    assert version_handle.version_id == "model_version_id"
    assert version_handle.model_id == "model_id"
    api.create_model_version_from_truss.assert_called_once()
    _, kwargs = api.create_model_version_from_truss.call_args
    for k, v in inputs.items():
        assert kwargs[k] == v
    assert kwargs.get("deploy_timeout_minutes") is None


@pytest.mark.parametrize("allow_truss_download", [True, False])
@pytest.mark.parametrize("is_draft", [True, False])
def test_create_truss_service_handles_allow_truss_download_for_new_models(
    is_draft, allow_truss_download
):
    api = MagicMock()
    return_value = {
        "id": "model_version_id",
        "oracle": {"id": "model_id", "hostname": "hostname"},
    }
    api.create_model_from_truss.return_value = return_value
    api.create_development_model_from_truss.return_value = return_value

    version_handle = create_truss_service(
        api,
        "model_name",
        "s3_key",
        "config",
        b10_types.TrussUserEnv.collect(),
        preserve_previous_prod_deployment=False,
        is_draft=is_draft,
        model_id=None,
        deployment_name="deployment_name",
        allow_truss_download=allow_truss_download,
    )
    assert version_handle.version_id == "model_version_id"
    assert version_handle.model_id == "model_id"

    create_model_mock = (
        api.create_development_model_from_truss
        if is_draft
        else api.create_model_from_truss
    )
    create_model_mock.assert_called_once()
    _, kwargs = create_model_mock.call_args
    assert kwargs["allow_truss_download"] is allow_truss_download


def test_validate_truss_config():
    def mock_validate_truss(config):
        if config == {}:
            return {"success": True, "details": json.dumps({})}
        elif "hi" in config:
            return {"success": False, "details": json.dumps({"errors": ["error"]})}
        else:
            return {
                "success": False,
                "details": json.dumps({"errors": ["error", "and another one"]}),
            }

    api = MagicMock()
    api.validate_truss.side_effect = mock_validate_truss

    assert core.validate_truss_config_against_backend(api, {}) is None
    with pytest.raises(
        ValidationError, match="Validation failed with the following errors:\n  error"
    ):
        core.validate_truss_config_against_backend(api, {"hi": "hi"})
    with pytest.raises(
        ValidationError,
        match="Validation failed with the following errors:\n  error\n  and another one",
    ):
        core.validate_truss_config_against_backend(api, {"should_error": "hi"})


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

    # Mock the _fetch_log_batch method directly
    mock_fetch = mock.Mock(side_effect=[batch1_logs, batch2_logs, batch3_logs, []])
    baseten_api._fetch_log_batch = mock_fetch

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
    assert mock_fetch.call_count == 4  # 3 batches + 1 empty batch to stop

    # Verify first call parameters
    first_call_args = mock_fetch.call_args_list[0]
    assert first_call_args[0][0] == "project-123"  # project_id
    assert first_call_args[0][1] == "job-456"  # job_id

    # Verify the query body contains expected parameters
    query_params = first_call_args[0][2]  # query_params
    assert query_params["limit"] == 2
    assert query_params["direction"] == "asc"
    assert "start_epoch_millis" in query_params
    assert "end_epoch_millis" in query_params


def test_get_training_job_logs_with_pagination_empty_response(baseten_api):
    """Test pagination when no logs are returned"""
    # Mock the _fetch_log_batch method directly
    mock_fetch = mock.Mock(return_value=[])
    baseten_api._fetch_log_batch = mock_fetch

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
    mock_fetch.assert_called_once()


def test_get_training_job_logs_with_pagination_partial_batch(baseten_api):
    """Test pagination when the last batch has fewer logs than batch_size"""
    batch1_logs = [
        {"timestamp": "1640995200000000000", "message": "Log 1"},
        {"timestamp": "1640995260000000000", "message": "Log 2"},
    ]
    batch2_logs = [{"timestamp": "1640995320000000000", "message": "Log 3"}]

    # Mock the _fetch_log_batch method directly
    mock_fetch = mock.Mock(side_effect=[batch1_logs, batch2_logs, []])
    baseten_api._fetch_log_batch = mock_fetch

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

    # Verify only 3 API calls (2 batches + 1 empty batch to stop)
    assert mock_fetch.call_count == 3


def test_get_training_job_logs_with_pagination_max_iterations(baseten_api):
    """Test pagination when maximum iterations are reached"""
    # Mock logs that would cause infinite pagination
    batch_logs = [
        {"timestamp": "1640995200000000000", "message": "Log 1"},
        {"timestamp": "1640995260000000000", "message": "Log 2"},
    ]

    # Mock the _fetch_log_batch method directly
    # Configure mock to always return the same batch (simulating infinite pagination)
    mock_fetch = mock.Mock(return_value=batch_logs)
    baseten_api._fetch_log_batch = mock_fetch

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
    assert mock_fetch.call_count == MAX_ITERATIONS


def test_get_training_job_logs_with_pagination_api_error(baseten_api):
    """Test pagination when API returns an error"""
    # Mock the _fetch_log_batch method directly
    # Configure mock to raise an exception
    mock_fetch = mock.Mock(side_effect=Exception("API Error"))
    baseten_api._fetch_log_batch = mock_fetch

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
    mock_fetch.assert_called_once()


def test_get_training_job_logs_with_pagination_custom_batch_size(baseten_api):
    """Test pagination with custom batch size"""
    mock_logs = [
        {"timestamp": "1640995200000000000", "message": "Log 1"},
        {"timestamp": "1640995260000000000", "message": "Log 2"},
    ]

    # Mock the _fetch_log_batch method directly
    mock_fetch = mock.Mock(side_effect=[mock_logs, []])
    baseten_api._fetch_log_batch = mock_fetch

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
    call_args = mock_fetch.call_args
    query_params = call_args[0][2]  # query_params
    assert query_params["limit"] == 50


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

    # Mock the _fetch_log_batch method directly
    mock_fetch = mock.Mock(
        side_effect=[mock_logs_batch_1, mock_logs_batch_2, mock_logs_batch_3, []]
    )
    baseten_api._fetch_log_batch = mock_fetch

    # Mock get_training_job method
    baseten_api.get_training_job = mock.Mock(
        return_value={"training_job": {"created_at": "2022-01-01T00:00:00Z"}}
    )

    result = get_training_job_logs_with_pagination(
        baseten_api, "project-123", "job-456", batch_size=2
    )

    # Verify the result
    assert result == mock_logs_batch_1 + mock_logs_batch_2 + mock_logs_batch_3
    assert mock_fetch.call_count == 4  # 3 batches + 1 empty batch to stop


def test_get_training_job_logs_with_pagination_timestamp_conversion(baseten_api):
    """Test that timestamp conversion from nanoseconds to milliseconds works correctly"""
    batch1_logs = [
        {"timestamp": "1640995200000000000", "message": "Log 1"}  # 1640995200000 ms
    ]
    batch2_logs = [
        {"timestamp": "1640995260000000000", "message": "Log 2"}  # 1640995260000 ms
    ]

    # Mock the _fetch_log_batch method directly
    mock_fetch = mock.Mock(side_effect=[batch1_logs, batch2_logs, []])
    baseten_api._fetch_log_batch = mock_fetch

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
    second_call = mock_fetch.call_args_list[1]
    query_params = second_call[0][2]  # query_params
    # Should be 1640995200000 + 1 = 1640995200001
    assert query_params["start_epoch_millis"] == 1640995200001


def test_get_training_job_logs_with_pagination_query_body_filtering(baseten_api):
    """Test that None values are properly filtered from query body"""
    mock_logs = [{"timestamp": "1640995200000000000", "message": "Log 1"}]

    # Mock the _fetch_log_batch method directly
    mock_fetch = mock.Mock(side_effect=[mock_logs, []])
    baseten_api._fetch_log_batch = mock_fetch

    # Mock get_training_job method
    baseten_api.get_training_job = mock.Mock(
        return_value={"training_job": {"created_at": "2022-01-01T00:00:00Z"}}
    )

    get_training_job_logs_with_pagination(baseten_api, "project-123", "job-456")

    # Verify the API call
    call_args = mock_fetch.call_args
    query_params = call_args[0][2]  # query_params

    # Verify that all required values are included in the query body
    assert "start_epoch_millis" in query_params
    assert "end_epoch_millis" in query_params
    assert "limit" in query_params
    assert "direction" in query_params


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

    # Mock the _fetch_log_batch method directly
    # First call fails with 500, second call succeeds
    mock_response_500 = mock.Mock()
    mock_response_500.status_code = 500
    mock_error_500 = requests.HTTPError("Server Error")
    mock_error_500.response = mock_response_500

    mock_fetch = mock.Mock(
        side_effect=[
            mock_error_500,  # First call fails
            batch_logs,  # Second call succeeds
        ]
    )
    baseten_api._fetch_log_batch = mock_fetch

    # Mock get_training_job method
    baseten_api.get_training_job = mock.Mock(
        return_value={"training_job": {"created_at": "2022-01-01T00:00:00Z"}}
    )

    result = get_training_job_logs_with_pagination(
        baseten_api, "project-123", "job-456", batch_size=1000
    )

    # Should get the logs after retry
    assert result == batch_logs

    # Should have made 3 calls (first fails, retry with reduced batch size, then succeeds)
    assert mock_fetch.call_count == 3


def test_get_training_job_logs_with_pagination_non_server_error(baseten_api):
    """Test pagination with non-server error (should not retry)"""
    # Mock the _fetch_log_batch method directly

    # Mock a 400 error (client error, not server error)
    mock_response_400 = mock.Mock()
    mock_response_400.status_code = 400
    mock_error_400 = requests.HTTPError("Bad Request")
    mock_error_400.response = mock_response_400

    mock_fetch = mock.Mock(side_effect=mock_error_400)
    baseten_api._fetch_log_batch = mock_fetch

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
    assert mock_fetch.call_count == 1


def test_get_training_job_logs_with_pagination_default_batch_size(baseten_api):
    """Test that default batch size is MAX_BATCH_SIZE"""
    mock_logs = [{"timestamp": "1640995200000000000", "message": "Log 1"}]

    # Mock the _fetch_log_batch method directly
    mock_fetch = mock.Mock(side_effect=[mock_logs, []])
    baseten_api._fetch_log_batch = mock_fetch

    # Mock get_training_job method
    baseten_api.get_training_job = mock.Mock(
        return_value={"training_job": {"created_at": "2022-01-01T00:00:00Z"}}
    )

    get_training_job_logs_with_pagination(baseten_api, "project-123", "job-456")

    # Verify the API call used default batch size
    call_args = mock_fetch.call_args
    query_params = call_args[0][2]  # query_params

    assert query_params["limit"] == MAX_BATCH_SIZE


def test_create_truss_service_passes_deploy_timeout_minutes():
    """Test that deploy_timeout_minutes is passed through to create_model_version_from_truss"""
    api = MagicMock()
    return_value = {
        "id": "model_version_id",
        "oracle": {"id": "model_id", "hostname": "hostname"},
        "instance_type": {"name": "1x2"},
    }
    api.create_model_version_from_truss.return_value = return_value
    version_handle = create_truss_service(
        api,
        "model_name",
        "s3_key",
        "config",
        b10_types.TrussUserEnv.collect(),
        is_draft=False,
        model_id="model_id",
        environment="staging",
        deploy_timeout_minutes=600,
    )

    assert version_handle.version_id == "model_version_id"
    assert version_handle.model_id == "model_id"
    api.create_model_version_from_truss.assert_called_once()
    _, kwargs = api.create_model_version_from_truss.call_args
    assert kwargs["deploy_timeout_minutes"] == 600


def test_create_truss_service_passes_deploy_timeout_minutes_with_other_params():
    """Test that deploy_timeout_minutes works correctly with other parameters like preserve_env_instance_type"""
    api = MagicMock()
    return_value = {
        "id": "model_version_id",
        "oracle": {"id": "model_id", "hostname": "hostname"},
        "instance_type": {"name": "1x2"},
    }
    api.create_model_version_from_truss.return_value = return_value
    version_handle = create_truss_service(
        api,
        "model_name",
        "s3_key",
        "config",
        b10_types.TrussUserEnv.collect(),
        is_draft=False,
        model_id="model_id",
        environment="production",
        preserve_env_instance_type=False,
        deploy_timeout_minutes=900,
    )

    assert version_handle.version_id == "model_version_id"
    api.create_model_version_from_truss.assert_called_once()
    _, kwargs = api.create_model_version_from_truss.call_args
    assert kwargs["deploy_timeout_minutes"] == 900
    assert kwargs["preserve_env_instance_type"] is False
    assert kwargs["environment"] == "production"


def test_create_truss_service_passes_deploy_timeout_minutes_for_development_model():
    """Test that deploy_timeout_minutes is passed through to create_development_model_from_truss"""
    api = MagicMock()
    return_value = {
        "id": "model_version_id",
        "oracle": {"id": "model_id", "hostname": "hostname"},
        "instance_type": {"name": "1x2"},
    }
    api.create_development_model_from_truss.return_value = return_value
    version_handle = create_truss_service(
        api,
        "model_name",
        "s3_key",
        "config",
        b10_types.TrussUserEnv.collect(),
        is_draft=True,
        model_id=None,
        deploy_timeout_minutes=600,
    )

    assert version_handle.version_id == "model_version_id"
    assert version_handle.model_id == "model_id"
    api.create_development_model_from_truss.assert_called_once()
    _, kwargs = api.create_development_model_from_truss.call_args
    assert kwargs["deploy_timeout_minutes"] == 600
