"""Tests for disable_truss_download parameter in public API push function."""

from unittest.mock import Mock, patch
import pathlib

import pytest

from truss_chains import public_api
from truss_chains.deployment.deployment_client import BasetenChainService


class TestPublicApiDisableTrussDownload:
    """Test disable_truss_download parameter in public API push function."""

    @pytest.fixture
    def mock_entrypoint(self):
        """Create a mock entrypoint chainlet class."""
        class MockEntrypoint:
            def __init__(self):
                self.__file__ = "/tmp/test_chain.py"
        
        return MockEntrypoint

    @pytest.fixture
    def mock_service(self):
        """Create a mock BasetenChainService."""
        # Create a mock that inherits from BasetenChainService
        class MockBasetenChainService(BasetenChainService):
            def __init__(self):
                # Call parent constructor with required parameters
                super().__init__(
                    name="test_chain",
                    chain_deployment_handle=Mock(),
                    remote=Mock(),
                )
            
            @property
            def run_remote_url(self) -> str:
                return "http://test.com/run_remote"
            
            @property
            def is_websocket(self) -> bool:
                return False
        
        return MockBasetenChainService()

    @patch("truss_chains.public_api.deployment_client.push")
    def test_push_with_disable_truss_download_true(self, mock_push, mock_entrypoint, mock_service):
        """Test that disable_truss_download=True is passed through to deployment client."""
        mock_push.return_value = mock_service
        
        # Call push with disable_truss_download=True
        result = public_api.push(
            entrypoint=mock_entrypoint,
            chain_name="test_chain",
            disable_truss_download=True,
        )
        
        # Verify that deployment_client.push was called
        mock_push.assert_called_once()
        call_args = mock_push.call_args
        options = call_args[0][1]  # Second argument is the options
        
        # Check that disable_truss_download is set to True in the options
        assert hasattr(options, 'disable_truss_download')
        assert options.disable_truss_download is True
        
        # Verify the result
        assert result == mock_service

    @patch("truss_chains.public_api.deployment_client.push")
    def test_push_with_disable_truss_download_false(self, mock_push, mock_entrypoint, mock_service):
        """Test that disable_truss_download=False is passed through to deployment client."""
        mock_push.return_value = mock_service
        
        # Call push with disable_truss_download=False
        result = public_api.push(
            entrypoint=mock_entrypoint,
            chain_name="test_chain",
            disable_truss_download=False,
        )
        
        # Verify that deployment_client.push was called
        mock_push.assert_called_once()
        call_args = mock_push.call_args
        options = call_args[0][1]  # Second argument is the options
        
        # Check that disable_truss_download is set to False in the options
        assert hasattr(options, 'disable_truss_download')
        assert options.disable_truss_download is False
        
        # Verify the result
        assert result == mock_service

    @patch("truss_chains.public_api.deployment_client.push")
    def test_push_without_disable_truss_download_defaults_to_false(self, mock_push, mock_entrypoint, mock_service):
        """Test that disable_truss_download defaults to False when not specified."""
        mock_push.return_value = mock_service
        
        # Call push without disable_truss_download parameter
        result = public_api.push(
            entrypoint=mock_entrypoint,
            chain_name="test_chain",
        )
        
        # Verify that deployment_client.push was called
        mock_push.assert_called_once()
        call_args = mock_push.call_args
        options = call_args[0][1]  # Second argument is the options
        
        # Check that disable_truss_download defaults to False
        assert hasattr(options, 'disable_truss_download')
        assert options.disable_truss_download is False
        
        # Verify the result
        assert result == mock_service

    @patch("truss_chains.public_api.deployment_client.push")
    def test_push_with_all_parameters_including_disable_truss_download(self, mock_push, mock_entrypoint, mock_service):
        """Test that all parameters including disable_truss_download are passed correctly."""
        mock_push.return_value = mock_service
        
        # Call push with all parameters
        result = public_api.push(
            entrypoint=mock_entrypoint,
            chain_name="test_chain",
            publish=True,
            promote=False,
            only_generate_trusses=False,
            remote="test_remote",
            environment="test_env",
            include_git_info=True,
            disable_truss_download=True,
        )
        
        # Verify that deployment_client.push was called
        mock_push.assert_called_once()
        call_args = mock_push.call_args
        options = call_args[0][1]  # Second argument is the options
        
        # Check all parameters
        assert options.chain_name == "test_chain"
        assert options.publish is True
        assert options.remote == "test_remote"
        assert options.environment == "test_env"
        assert options.include_git_info is True
        assert options.disable_truss_download is True
        
        # Verify the result
        assert result == mock_service

    def test_push_options_baseten_create_called_with_correct_parameters(self, mock_entrypoint, mock_service):
        """Test that PushOptionsBaseten.create is called with the correct parameters."""
        with patch("truss_chains.public_api.deployment_client.push") as mock_push:
            with patch("truss_chains.public_api.private_types.PushOptionsBaseten.create") as mock_create:
                mock_push.return_value = mock_service
                mock_create.return_value = Mock()
                
                # Call push with disable_truss_download=True
                public_api.push(
                    entrypoint=mock_entrypoint,
                    chain_name="test_chain",
                    disable_truss_download=True,
                )
                
                # Verify that PushOptionsBaseten.create was called with correct parameters
                mock_create.assert_called_once()
                call_args = mock_create.call_args
                assert call_args[1]["chain_name"] == "test_chain"
                assert call_args[1]["disable_truss_download"] is True
                assert call_args[1]["publish"] is True  # default value
                assert call_args[1]["promote"] is True  # default value
