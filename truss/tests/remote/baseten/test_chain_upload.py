import tempfile
import pathlib
from unittest.mock import Mock, patch, MagicMock
import pytest
from io import BytesIO

from truss.remote.baseten.api import BasetenApi
from truss.remote.baseten.core import upload_chain_artifact
from truss.remote.baseten.remote import BasetenRemote
from truss.remote.baseten import custom_types as b10_types


class TestChainUpload:
    """Test chain artifact upload functionality."""

    def test_chain_s3_upload_credentials_api_method(self):
        """Test that the chain_s3_upload_credentials API method works correctly."""
        # Mock the API response
        mock_response = {
            "data": {
                "chain_s3_upload_credentials": {
                    "s3_bucket": "test-chain-bucket",
                    "s3_key": "chains/test-uuid/chain.tgz",
                    "aws_access_key_id": "test_access_key",
                    "aws_secret_access_key": "test_secret_key",
                    "aws_session_token": "test_session_token"
                }
            }
        }

        # Create a mock API instance
        api = Mock(spec=BasetenApi)
        api._post_graphql_query.return_value = mock_response

        # Call the method
        result = api.chain_s3_upload_credentials()

        # Verify the result
        assert result == mock_response["data"]["chain_s3_upload_credentials"]
        assert result["s3_bucket"] == "test-chain-bucket"
        assert result["s3_key"] == "chains/test-uuid/chain.tgz"
        assert result["aws_access_key_id"] == "test_access_key"

        # Verify the GraphQL query was called
        api._post_graphql_query.assert_called_once()
        query = api._post_graphql_query.call_args[0][0]
        assert "chain_s3_upload_credentials" in query
        assert "s3_bucket" in query
        assert "s3_key" in query

    @patch('truss.remote.baseten.core.multipart_upload_boto3')
    def test_upload_chain_artifact_function(self, mock_multipart_upload):
        """Test the upload_chain_artifact function."""
        # Mock API response
        mock_credentials = {
            "s3_bucket": "test-chain-bucket",
            "s3_key": "chains/test-uuid/chain.tgz",
            "aws_access_key_id": "test_access_key",
            "aws_secret_access_key": "test_secret_key",
            "aws_session_token": "test_session_token"
        }

        # Create mock API
        api = Mock(spec=BasetenApi)
        api.chain_s3_upload_credentials.return_value = mock_credentials

        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix=".tgz", delete=False) as temp_file:
            temp_file.write(b"test chain content")
            temp_file.flush()

            # Call the function
            result = upload_chain_artifact(api, temp_file, None)

            # Verify the result
            assert result == "chains/test-uuid/chain.tgz"

            # Verify the API was called
            api.chain_s3_upload_credentials.assert_called_once()

            # Verify multipart upload was called with correct parameters
            mock_multipart_upload.assert_called_once()
            call_args = mock_multipart_upload.call_args
            assert call_args[0][0] == temp_file.name  # file path
            assert call_args[0][1] == "test-chain-bucket"  # bucket
            assert call_args[0][2] == "chains/test-uuid/chain.tgz"  # key
            assert call_args[0][3] == {  # credentials
                "aws_access_key_id": "test_access_key",
                "aws_secret_access_key": "test_secret_key",
                "aws_session_token": "test_session_token"
            }

    @patch('truss.remote.baseten.remote.upload_chain_artifact')
    @patch('truss.remote.baseten.remote.create_tar_with_progress_bar')
    @patch('truss.remote.baseten.remote.create_chain_atomic')
    def test_push_chain_atomic_with_chain_upload(
        self,
        mock_create_chain_atomic,
        mock_create_tar,
        mock_upload_chain_artifact
    ):
        """Test that push_chain_atomic uploads raw chain artifact when chain_root is provided."""
        # Setup mocks
        mock_create_chain_atomic.return_value = Mock()
        mock_create_tar.return_value = Mock()
        mock_upload_chain_artifact.return_value = "chains/test-uuid/chain.tgz"

        # Create mock API
        api = Mock(spec=BasetenApi)

        # Create mock remote
        remote = BasetenRemote(api, Mock())

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

        with patch.object(remote, '_prepare_push', return_value=mock_push_data):
            with patch('truss.remote.baseten.remote.truss_build.load') as mock_load:
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
                    publish=True
                )

                # Verify chain artifact upload was called
                mock_create_tar.assert_called_once_with(
                    source_dir=chain_root,
                    ignore_patterns=None,
                    delete=True,
                    progress_bar=None
                )
                mock_upload_chain_artifact.assert_called_once()

                # Verify create_chain_atomic was called
                mock_create_chain_atomic.assert_called_once()

    @patch('truss.remote.baseten.remote.create_chain_atomic')
    def test_push_chain_atomic_without_chain_upload(self, mock_create_chain_atomic):
        """Test that push_chain_atomic skips chain upload when chain_root is None."""
        # Setup mocks
        mock_create_chain_atomic.return_value = Mock()

        # Create mock API
        api = Mock(spec=BasetenApi)

        # Create mock remote
        remote = BasetenRemote(api, Mock())

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

        with patch.object(remote, '_prepare_push', return_value=mock_push_data):
            with patch('truss.remote.baseten.remote.truss_build.load') as mock_load:
                mock_truss_handle = Mock()
                mock_truss_handle.spec.config.model_name = "test-model"
                mock_load.return_value = mock_truss_handle

                with patch('truss.remote.baseten.remote.upload_chain_artifact') as mock_upload:
                    with patch('truss.remote.baseten.remote.create_tar_with_progress_bar') as mock_tar:
                        # Call push_chain_atomic without chain_root
                        result = remote.push_chain_atomic(
                            chain_name=chain_name,
                            entrypoint_artifact=entrypoint_artifact,
                            dependency_artifacts=dependency_artifacts,
                            truss_user_env=truss_user_env,
                            chain_root=None,  # No chain root
                            publish=True
                        )

                        # Verify chain artifact upload was NOT called
                        mock_tar.assert_not_called()
                        mock_upload.assert_not_called()

                        # Verify create_chain_atomic was called
                        mock_create_chain_atomic.assert_called_once()

    def test_chain_s3_upload_credentials_query_structure(self):
        """Test that the GraphQL query has the correct structure."""
        api = Mock(spec=BasetenApi)
        api._post_graphql_query.return_value = {
            "data": {"chain_s3_upload_credentials": {}}
        }

        # Call the method
        api.chain_s3_upload_credentials()

        # Get the query that was sent
        query = api._post_graphql_query.call_args[0][0]

        # Verify query structure
        assert "query" in query or query.strip().startswith("{")
        assert "chain_s3_upload_credentials" in query
        assert "s3_bucket" in query
        assert "s3_key" in query
        assert "aws_access_key_id" in query
        assert "aws_secret_access_key" in query
        assert "aws_session_token" in query

    @patch('truss.remote.baseten.core.multipart_upload_boto3')
    def test_upload_chain_artifact_error_handling(self, mock_multipart_upload):
        """Test error handling in upload_chain_artifact."""
        # Mock API to raise an exception
        api = Mock(spec=BasetenApi)
        api.chain_s3_upload_credentials.side_effect = Exception("API Error")

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
            "aws_access_key_id": "access_key",
            "aws_secret_access_key": "secret_key",
            "aws_session_token": "session_token",
            "extra_field": "should_be_ignored"
        }

        api = Mock(spec=BasetenApi)
        api.chain_s3_upload_credentials.return_value = mock_credentials

        with patch('truss.remote.baseten.core.multipart_upload_boto3') as mock_upload:
            with tempfile.NamedTemporaryFile(suffix=".tgz") as temp_file:
                upload_chain_artifact(api, temp_file, None)

                # Verify multipart upload was called with correct credentials
                call_args = mock_upload.call_args
                credentials = call_args[0][3]

                # Should only contain the AWS credentials, not extra fields
                assert credentials == {
                    "aws_access_key_id": "access_key",
                    "aws_secret_access_key": "secret_key",
                    "aws_session_token": "session_token"
                }
                assert "extra_field" not in credentials
