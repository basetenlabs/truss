import pathlib
import tempfile
from unittest.mock import Mock, patch

import pytest

from truss.remote.baseten import custom_types as b10_types
from truss.remote.baseten.api import BasetenApi
from truss.remote.baseten.core import upload_chain_artifact
from truss.remote.baseten.remote import BasetenRemote


class TestChainUpload:
    """Test chain artifact upload functionality."""

    def test_get_blob_credentials_for_chain(self):
        """Test that get_blob_credentials works correctly for chain blob type using GraphQL."""
        # Mock the GraphQL response
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

            # Call the method
            result = api.get_chain_s3_upload_credentials()

            # Verify the result
            assert result["s3_bucket"] == "test-chain-bucket"
            assert result["s3_key"] == "chains/test-uuid/chain.tgz"
            assert result["creds"]["aws_access_key_id"] == "test_access_key"
            assert result["creds"]["aws_secret_access_key"] == "test_secret_key"
            assert result["creds"]["aws_session_token"] == "test_session_token"

            # Verify the GraphQL call was made
            mock_graphql.assert_called_once()
            call_args = mock_graphql.call_args
            assert "chain_s3_upload_credentials" in call_args[0][0]

    def test_get_blob_credentials_for_other_types_uses_rest(self):
        """Test that get_blob_credentials uses REST API for non-chain blob types."""
        # Mock the REST API response
        mock_response = {
            "s3_bucket": "test-bucket",
            "s3_key": "test-key",
            "creds": {
                "aws_access_key_id": "test_access_key",
                "aws_secret_access_key": "test_secret_key",
                "aws_session_token": "test_session_token",
            },
        }

        # Create a real API instance and mock both REST API and GraphQL calls
        mock_auth_service = Mock()
        mock_auth_service.authenticate.return_value = Mock(value="test-token")
        api = BasetenApi("https://test.baseten.co", mock_auth_service)
        with patch.object(api, "_rest_api_client") as mock_client, patch.object(
            api, "_post_graphql_query"
        ) as mock_graphql:
            mock_client.get.return_value = mock_response

            # Call the method for model blob type
            result = api.get_blob_credentials(b10_types.BlobType.MODEL)

            # Verify the result
            assert result["s3_bucket"] == "test-bucket"
            assert result["s3_key"] == "test-key"

            # Verify the REST API call was made, not GraphQL
            mock_client.get.assert_called_once_with("v1/blobs/credentials/model")
            mock_graphql.assert_not_called()

    @patch("truss.remote.baseten.core.multipart_upload_boto3")
    def test_upload_chain_artifact_function(self, mock_multipart_upload):
        """Test the upload_chain_artifact function."""
        # Mock API response
        mock_credentials = {
            "s3_bucket": "test-chain-bucket",
            "s3_key": "chains/test-uuid/chain.tgz",
            "creds": {
                "aws_access_key_id": "test_access_key",
                "aws_secret_access_key": "test_secret_key",
                "aws_session_token": "test_session_token",
            },
        }

        # Create mock API
        api = Mock(spec=BasetenApi)
        api.get_chain_s3_upload_credentials.return_value = mock_credentials

        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix=".tgz", delete=False) as temp_file:
            temp_file.write(b"test chain content")
            temp_file.flush()

            # Call the function
            result = upload_chain_artifact(api, temp_file, None)

            # Verify the result
            assert result == "chains/test-uuid/chain.tgz"

            # Verify the API was called
            api.get_chain_s3_upload_credentials.assert_called_once_with()

            # Verify multipart upload was called with correct parameters
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
        self, mock_create_chain_atomic, mock_archive_dir, mock_upload_chain_artifact
    ):
        """Test that push_chain_atomic uploads raw chain artifact when chain_root is provided."""
        # Setup mocks
        mock_create_chain_atomic.return_value = Mock()
        mock_archive_dir.return_value = Mock()
        mock_upload_chain_artifact.return_value = "chains/test-uuid/chain.tgz"

        # Create mock API
        api = Mock(spec=BasetenApi)

        # Create mock remote with proper URL
        remote = BasetenRemote("https://test.baseten.co", "test-api-key")
        remote._api = api

        # Create test data
        chain_name = "test-chain"
        entrypoint_artifact = Mock()
        entrypoint_artifact.truss_dir = "/path/to/truss"
        entrypoint_artifact.display_name = "entrypoint"

        dependency_artifacts = []
        truss_user_env = Mock()
        chain_root = pathlib.Path("/path/to/chain")

        # Mock the _prepare_push method
        mock_push_data = Mock()
        mock_push_data.model_name = "test-model"
        mock_push_data.s3_key = "models/test-key"
        mock_push_data.encoded_config_str = "encoded_config"
        mock_push_data.is_draft = False
        mock_push_data.model_id = "model-id"
        mock_push_data.version_name = None

        with patch.object(remote, "_prepare_push", return_value=mock_push_data):
            with patch("truss.remote.baseten.remote.truss_build.load") as mock_load:
                mock_truss_handle = Mock()
                mock_truss_handle.spec.config.model_name = "test-model"
                mock_load.return_value = mock_truss_handle

                # Call push_chain_atomic with chain_root
                result = remote.push_chain_atomic(
                    chain_name=chain_name,
                    entrypoint_artifact=entrypoint_artifact,
                    dependency_artifacts=dependency_artifacts,
                    truss_user_env=truss_user_env,
                    chain_root=chain_root,
                    publish=True,
                )

                # Verify chain artifact upload was called
                mock_archive_dir.assert_called_once_with(
                    dir=chain_root, progress_bar=None
                )
                mock_upload_chain_artifact.assert_called_once()

                # Verify create_chain_atomic was called
                mock_create_chain_atomic.assert_called_once()

    @patch("truss.remote.baseten.remote.create_chain_atomic")
    def test_push_chain_atomic_without_chain_upload(self, mock_create_chain_atomic):
        """Test that push_chain_atomic skips chain upload when chain_root is None."""
        # Setup mocks
        mock_create_chain_atomic.return_value = Mock()

        # Create mock API
        api = Mock(spec=BasetenApi)

        # Create mock remote with proper URL
        remote = BasetenRemote("https://test.baseten.co", "test-api-key")
        remote._api = api

        # Create test data
        chain_name = "test-chain"
        entrypoint_artifact = Mock()
        entrypoint_artifact.truss_dir = "/path/to/truss"
        entrypoint_artifact.display_name = "entrypoint"

        dependency_artifacts = []
        truss_user_env = Mock()

        # Mock the _prepare_push method
        mock_push_data = Mock()
        mock_push_data.model_name = "test-model"
        mock_push_data.s3_key = "models/test-key"
        mock_push_data.encoded_config_str = "encoded_config"
        mock_push_data.is_draft = False
        mock_push_data.model_id = "model-id"
        mock_push_data.version_name = None

        with patch.object(remote, "_prepare_push", return_value=mock_push_data):
            with patch("truss.remote.baseten.remote.truss_build.load") as mock_load:
                mock_truss_handle = Mock()
                mock_truss_handle.spec.config.model_name = "test-model"
                mock_load.return_value = mock_truss_handle

                with patch(
                    "truss.remote.baseten.remote.upload_chain_artifact"
                ) as mock_upload:
                    with patch(
                        "truss.remote.baseten.remote.create_tar_with_progress_bar"
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

                        # Verify chain artifact upload was NOT called
                        mock_tar.assert_not_called()
                        mock_upload.assert_not_called()

                        # Verify create_chain_atomic was called
                        mock_create_chain_atomic.assert_called_once()

    @patch("truss.remote.baseten.core.multipart_upload_boto3")
    def test_upload_chain_artifact_error_handling(self, mock_multipart_upload):
        """Test error handling in upload_chain_artifact."""
        # Mock API to raise an exception
        api = Mock(spec=BasetenApi)
        api.get_chain_s3_upload_credentials.side_effect = Exception("API Error")

        with tempfile.NamedTemporaryFile(suffix=".tgz") as temp_file:
            # Should raise the exception
            with pytest.raises(Exception, match="API Error"):
                upload_chain_artifact(api, temp_file, None)

    def test_upload_chain_artifact_credentials_extraction(self):
        """Test that credentials are properly extracted from API response."""
        # Mock API response with extra fields
        mock_credentials = {
            "s3_bucket": "test-bucket",
            "s3_key": "chains/test-uuid/chain.tgz",
            "creds": {
                "aws_access_key_id": "access_key",
                "aws_secret_access_key": "secret_key",
                "aws_session_token": "session_token",
            },
            "extra_field": "should_be_ignored",
        }

        api = Mock(spec=BasetenApi)
        api.get_chain_s3_upload_credentials.return_value = mock_credentials

        with patch("truss.remote.baseten.core.multipart_upload_boto3") as mock_upload:
            with tempfile.NamedTemporaryFile(suffix=".tgz") as temp_file:
                upload_chain_artifact(api, temp_file, None)

                # Verify multipart upload was called with correct credentials
                call_args = mock_upload.call_args
                credentials = call_args[0][3]

                # Should only contain the AWS credentials, not extra fields
                assert credentials == {
                    "aws_access_key_id": "access_key",
                    "aws_secret_access_key": "secret_key",
                    "aws_session_token": "session_token",
                }
                assert "extra_field" not in credentials
