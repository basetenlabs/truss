import pathlib
import tempfile
from unittest.mock import Mock, patch

import pytest

from truss.remote.baseten import custom_types as b10_types
from truss.remote.baseten.api import BasetenApi
from truss.remote.baseten.core import upload_chain_artifact
from truss.remote.baseten.remote import BasetenRemote


@pytest.fixture
def mock_push_data():
    """Fixture providing mock push data for tests."""
    mock_push_data = Mock()
    mock_push_data.model_name = "test-model"
    mock_push_data.s3_key = "models/test-key"
    mock_push_data.encoded_config_str = "encoded_config"
    mock_push_data.is_draft = False
    mock_push_data.model_id = "model-id"
    mock_push_data.version_name = None
    return mock_push_data


@pytest.fixture
def mock_remote_context():
    """Fixture providing mock remote and context managers for tests."""
    api = Mock(spec=BasetenApi)

    remote = BasetenRemote("https://test.baseten.co", "test-api-key")
    remote._api = api

    chain_name = "test-chain"
    entrypoint_artifact = Mock()
    entrypoint_artifact.truss_dir = "/path/to/truss"
    entrypoint_artifact.display_name = "entrypoint"

    dependency_artifacts = []
    truss_user_env = Mock()
    chain_root = pathlib.Path("/path/to/chain")

    with patch.object(remote, "_prepare_push") as mock_prepare_push:
        with patch("truss.remote.baseten.remote.truss_build.load") as mock_load:
            mock_truss_handle = Mock()
            mock_truss_handle.spec.config.model_name = "test-model"
            mock_load.return_value = mock_truss_handle

            yield {
                "remote": remote,
                "api": api,
                "chain_name": chain_name,
                "entrypoint_artifact": entrypoint_artifact,
                "dependency_artifacts": dependency_artifacts,
                "truss_user_env": truss_user_env,
                "chain_root": chain_root,
                "mock_prepare_push": mock_prepare_push,
                "mock_load": mock_load,
                "mock_truss_handle": mock_truss_handle,
            }


def test_get_blob_credentials_for_chain():
    """Test that get_blob_credentials works correctly for chain blob type using GraphQL."""
    mock_graphql_response = {
        "data": {
            "chain_s3_upload_credentials": {
                "s3_bucket": "test-chain-bucket",
                "s3_key": "chains/test-uuid/chain.tgz",
                "aws_access_key_id": "test_access_key",
                "aws_secret_access_key": "test_secret_key",
                "aws_session_token": "test_session_token",
            }
        }
    }

    # Create a real API instance and mock the GraphQL call
    mock_auth_service = Mock()
    mock_auth_service.authenticate.return_value = Mock(value="test-token")
    api = BasetenApi("https://test.baseten.co", mock_auth_service)
    with patch.object(api, "_post_graphql_query") as mock_graphql:
        mock_graphql.return_value = mock_graphql_response

        result = api.get_chain_s3_upload_credentials()

        assert result.s3_bucket == "test-chain-bucket"
        assert result.s3_key == "chains/test-uuid/chain.tgz"
        assert result.aws_access_key_id == "test_access_key"
        assert result.aws_secret_access_key == "test_secret_key"
        assert result.aws_session_token == "test_session_token"

        mock_graphql.assert_called_once()
        call_args = mock_graphql.call_args
        assert "chain_s3_upload_credentials" in call_args[0][0]


def test_get_blob_credentials_for_other_types_uses_rest():
    """Test that get_blob_credentials uses REST API for non-chain blob types."""
    mock_response = {
        "s3_bucket": "test-bucket",
        "s3_key": "test-key",
        "creds": {
            "aws_access_key_id": "test_access_key",
            "aws_secret_access_key": "test_secret_key",
            "aws_session_token": "test_session_token",
        },
    }

    mock_auth_service = Mock()
    mock_auth_service.authenticate.return_value = Mock(value="test-token")
    api = BasetenApi("https://test.baseten.co", mock_auth_service)
    with (
        patch.object(api, "_rest_api_client") as mock_client,
        patch.object(api, "_post_graphql_query") as mock_graphql,
    ):
        mock_client.get.return_value = mock_response

        result = api.get_blob_credentials(b10_types.BlobType.MODEL)

        assert result["s3_bucket"] == "test-bucket"
        assert result["s3_key"] == "test-key"

        mock_client.get.assert_called_once_with("v1/blobs/credentials/model")
        mock_graphql.assert_not_called()


@patch("truss.remote.baseten.core.multipart_upload_boto3")
def test_upload_chain_artifact_function(mock_multipart_upload):
    """Test the upload_chain_artifact function."""
    # Mock ChainUploadCredentials object
    mock_credentials = Mock()
    mock_credentials.s3_bucket = "test-chain-bucket"
    mock_credentials.s3_key = "chains/test-uuid/chain.tgz"
    mock_credentials.aws_credentials = Mock()
    mock_credentials.aws_credentials.model_dump.return_value = {
        "aws_access_key_id": "test_access_key",
        "aws_secret_access_key": "test_secret_key",
        "aws_session_token": "test_session_token",
    }

    api = Mock(spec=BasetenApi)
    api.get_chain_s3_upload_credentials.return_value = mock_credentials

    with tempfile.NamedTemporaryFile(suffix=".tgz", delete=False) as temp_file:
        temp_file.write(b"test chain content")
        temp_file.flush()

        result = upload_chain_artifact(api, temp_file, None)

        assert result == "chains/test-uuid/chain.tgz"

        api.get_chain_s3_upload_credentials.assert_called_once_with()

        mock_multipart_upload.assert_called_once()
        call_args = mock_multipart_upload.call_args
        assert call_args[0][0] == temp_file.name  # file path
        assert call_args[0][1] == "test-chain-bucket"  # bucket
        assert call_args[0][2] == "chains/test-uuid/chain.tgz"  # key
        assert call_args[0][3] == {  # credentials
            "aws_access_key_id": "test_access_key",
            "aws_secret_access_key": "test_secret_key",
            "aws_session_token": "test_session_token",
        }


@patch("truss.remote.baseten.remote.upload_chain_artifact")
@patch("truss.remote.baseten.remote.archive_dir")
@patch("truss.remote.baseten.remote.create_chain_atomic")
def test_push_chain_atomic_with_chain_upload(
    mock_create_chain_atomic,
    mock_archive_dir,
    mock_upload_chain_artifact,
    mock_push_data,
    mock_remote_context,
):
    """Test that push_chain_atomic uploads raw chain artifact when chain_root is provided."""
    mock_create_chain_atomic.return_value = Mock()
    mock_archive_dir.return_value = Mock()
    mock_upload_chain_artifact.return_value = "chains/test-uuid/chain.tgz"

    context = mock_remote_context
    remote = context["remote"]
    chain_name = context["chain_name"]
    entrypoint_artifact = context["entrypoint_artifact"]
    dependency_artifacts = context["dependency_artifacts"]
    truss_user_env = context["truss_user_env"]
    chain_root = context["chain_root"]

    context["mock_prepare_push"].return_value = mock_push_data
    deployment_name = "custom_deployment"

    result = remote.push_chain_atomic(
        chain_name=chain_name,
        entrypoint_artifact=entrypoint_artifact,
        dependency_artifacts=dependency_artifacts,
        truss_user_env=truss_user_env,
        chain_root=chain_root,
        publish=True,
        deployment_name=deployment_name,
    )
    assert result == mock_create_chain_atomic.return_value

    mock_archive_dir.assert_called_once_with(dir=chain_root, progress_bar=None)
    mock_upload_chain_artifact.assert_called_once()
    mock_create_chain_atomic.assert_called_once()
    create_kwargs = mock_create_chain_atomic.call_args.kwargs
    assert create_kwargs["deployment_name"] == deployment_name

    prepare_kwargs = context["mock_prepare_push"].call_args.kwargs
    assert prepare_kwargs["deployment_name"] == deployment_name


@patch("truss.remote.baseten.remote.create_chain_atomic")
def test_push_chain_atomic_without_chain_upload(
    mock_create_chain_atomic, mock_push_data, mock_remote_context
):
    """Test that push_chain_atomic skips chain upload when chain_root is None."""
    mock_create_chain_atomic.return_value = Mock()

    context = mock_remote_context
    remote = context["remote"]
    chain_name = context["chain_name"]
    entrypoint_artifact = context["entrypoint_artifact"]
    dependency_artifacts = context["dependency_artifacts"]
    truss_user_env = context["truss_user_env"]

    context["mock_prepare_push"].return_value = mock_push_data

    with patch("truss.remote.baseten.remote.upload_chain_artifact") as mock_upload:
        with patch(
            "truss.remote.baseten.core.create_tar_with_progress_bar"
        ) as mock_tar:
            # Call push_chain_atomic without chain_root
            result = remote.push_chain_atomic(
                chain_name=chain_name,
                entrypoint_artifact=entrypoint_artifact,
                dependency_artifacts=dependency_artifacts,
                truss_user_env=truss_user_env,
                chain_root=None,  # No chain root
                publish=True,
            )

            assert result
            # Verify chain artifact upload was NOT called
            mock_tar.assert_not_called()
            mock_upload.assert_not_called()

            mock_create_chain_atomic.assert_called_once()
            create_kwargs = mock_create_chain_atomic.call_args.kwargs
            assert "deployment_name" in create_kwargs
            assert create_kwargs["deployment_name"] is None


@patch("truss.remote.baseten.core.multipart_upload_boto3")
def test_upload_chain_artifact_error_handling(mock_multipart_upload):
    """Test error handling in upload_chain_artifact."""
    # Mock API to raise an exception
    api = Mock(spec=BasetenApi)
    api.get_chain_s3_upload_credentials.side_effect = Exception("API Error")

    with tempfile.NamedTemporaryFile(suffix=".tgz") as temp_file:
        # Should raise the exception
        with pytest.raises(Exception, match="API Error"):
            upload_chain_artifact(api, temp_file, None)


def test_upload_chain_artifact_credentials_extraction():
    """Test that credentials are properly extracted from API response."""
    # Mock ChainUploadCredentials object
    mock_credentials = Mock()
    mock_credentials.s3_bucket = "test-bucket"
    mock_credentials.s3_key = "chains/test-uuid/chain.tgz"
    mock_credentials.aws_credentials = Mock()
    mock_credentials.aws_credentials.model_dump.return_value = {
        "aws_access_key_id": "access_key",
        "aws_secret_access_key": "secret_key",
        "aws_session_token": "session_token",
    }

    api = Mock(spec=BasetenApi)
    api.get_chain_s3_upload_credentials.return_value = mock_credentials

    with patch("truss.remote.baseten.core.multipart_upload_boto3") as mock_upload:
        with tempfile.NamedTemporaryFile(suffix=".tgz") as temp_file:
            upload_chain_artifact(api, temp_file, None)

            call_args = mock_upload.call_args
            credentials = call_args[0][3]

            assert credentials == {
                "aws_access_key_id": "access_key",
                "aws_secret_access_key": "secret_key",
                "aws_session_token": "session_token",
            }
            assert "extra_field" not in credentials
