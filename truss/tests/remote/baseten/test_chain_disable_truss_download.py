"""Tests for disable_truss_download parameter in chain deployment."""

from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

import pytest

from truss.remote.baseten import remote as b10_remote
from truss.remote.baseten import custom_types as b10_types
from truss.remote.baseten.core import ChainDeploymentHandleAtomic


class TestChainDisableTrussDownload:
    """Test disable_truss_download parameter in chain deployment."""

    @pytest.fixture
    def mock_remote(self):
        """Create a BasetenRemote instance with mocked API."""
        remote = b10_remote.BasetenRemote("http://test.com", "test_key")
        remote._api = Mock()
        return remote

    @pytest.fixture
    def mock_chainlet_artifact(self):
        """Create a mock ChainletArtifact."""
        artifact = Mock(spec=b10_types.ChainletArtifact)
        artifact.truss_dir = Path("/tmp/test_truss")
        artifact.display_name = "test_chainlet"
        return artifact

    @pytest.fixture
    def mock_truss_handle(self):
        """Create a mock TrussHandle."""
        truss_handle = Mock()
        truss_handle.spec.config.model_name = "test_model"
        return truss_handle

    @pytest.fixture
    def mock_push_data(self):
        """Create a mock FinalPushData."""
        push_data = Mock()
        push_data.model_name = "test_model"
        push_data.s3_key = "test_s3_key"
        push_data.encoded_config_str = "test_config"
        push_data.is_draft = False
        push_data.model_id = None
        push_data.version_name = None
        push_data.allow_truss_download = True
        return push_data

    @patch("truss.remote.baseten.remote.create_chain_atomic")
    @patch("truss.remote.baseten.remote.truss_build.load")
    def test_push_chain_atomic_with_disable_truss_download_true(
        self, mock_load, mock_create_chain_atomic, mock_remote, mock_chainlet_artifact, mock_truss_handle, mock_push_data
    ):
        """Test that disable_truss_download=True is passed to _prepare_push."""
        mock_load.return_value = mock_truss_handle
        with patch.object(mock_remote, '_prepare_push', return_value=mock_push_data) as mock_prepare_push:
            # Mock the return value for create_chain_atomic
            mock_deployment_handle = ChainDeploymentHandleAtomic(
                chain_deployment_id="test_deployment_id",
                chain_id="test_chain_id",
                hostname="test_hostname",
                is_draft=False,
            )
            mock_create_chain_atomic.return_value = mock_deployment_handle
            
            # Call push_chain_atomic with disable_truss_download=True
            result = mock_remote.push_chain_atomic(
                chain_name="test_chain",
                entrypoint_artifact=mock_chainlet_artifact,
                dependency_artifacts=[],
                truss_user_env=b10_types.TrussUserEnv.collect(),
                publish=True,
                disable_truss_download=True,
            )
            
            # Verify that _prepare_push was called with disable_truss_download=True
            mock_prepare_push.assert_called()
            call_args = mock_prepare_push.call_args
            assert call_args[1]["disable_truss_download"] is True
            
            # Verify the result
            assert result == mock_deployment_handle

    @patch("truss.remote.baseten.remote.create_chain_atomic")
    @patch("truss.remote.baseten.remote.truss_build.load")
    def test_push_chain_atomic_with_disable_truss_download_false(
        self, mock_load, mock_create_chain_atomic, mock_remote, mock_chainlet_artifact, mock_truss_handle, mock_push_data
    ):
        """Test that disable_truss_download=False is passed to _prepare_push."""
        mock_load.return_value = mock_truss_handle
        with patch.object(mock_remote, '_prepare_push', return_value=mock_push_data) as mock_prepare_push:
            # Mock the return value for create_chain_atomic
            mock_deployment_handle = ChainDeploymentHandleAtomic(
                chain_deployment_id="test_deployment_id",
                chain_id="test_chain_id",
                hostname="test_hostname",
                is_draft=False,
            )
            mock_create_chain_atomic.return_value = mock_deployment_handle
            
            # Call push_chain_atomic with disable_truss_download=False
            result = mock_remote.push_chain_atomic(
                chain_name="test_chain",
                entrypoint_artifact=mock_chainlet_artifact,
                dependency_artifacts=[],
                truss_user_env=b10_types.TrussUserEnv.collect(),
                publish=True,
                disable_truss_download=False,
            )
            
            # Verify that _prepare_push was called with disable_truss_download=False
            mock_prepare_push.assert_called()
            call_args = mock_prepare_push.call_args
            assert call_args[1]["disable_truss_download"] is False
            
            # Verify the result
            assert result == mock_deployment_handle

    @patch("truss.remote.baseten.remote.create_chain_atomic")
    @patch("truss.remote.baseten.remote.truss_build.load")
    def test_push_chain_atomic_with_disable_truss_download_default(
        self, mock_load, mock_create_chain_atomic, mock_remote, mock_chainlet_artifact, mock_truss_handle, mock_push_data
    ):
        """Test that disable_truss_download defaults to False when not specified."""
        mock_load.return_value = mock_truss_handle
        with patch.object(mock_remote, '_prepare_push', return_value=mock_push_data) as mock_prepare_push:
            # Mock the return value for create_chain_atomic
            mock_deployment_handle = ChainDeploymentHandleAtomic(
                chain_deployment_id="test_deployment_id",
                chain_id="test_chain_id",
                hostname="test_hostname",
                is_draft=False,
            )
            mock_create_chain_atomic.return_value = mock_deployment_handle
            
            # Call push_chain_atomic without disable_truss_download parameter
            result = mock_remote.push_chain_atomic(
                chain_name="test_chain",
                entrypoint_artifact=mock_chainlet_artifact,
                dependency_artifacts=[],
                truss_user_env=b10_types.TrussUserEnv.collect(),
                publish=True,
            )
            
            # Verify that _prepare_push was called with disable_truss_download=False (default)
            mock_prepare_push.assert_called()
            call_args = mock_prepare_push.call_args
            assert call_args[1]["disable_truss_download"] is False
            
            # Verify the result
            assert result == mock_deployment_handle

