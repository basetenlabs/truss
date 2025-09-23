from unittest.mock import Mock, call, mock_open, patch

import pytest
import requests

from truss.cli.train.core import (
    _get_all_train_init_example_options,
    _get_train_init_example_info,
    download_git_directory,
)


class TestGetTrainInitExampleOptions:
    """Test cases for _get_train_init_example_options function"""

    @patch("requests.get")
    def test_successful_request_without_token(self, mock_get):
        """Test successful API call without authentication token"""
        # Arrange
        mock_response = Mock()
        mock_response.json.return_value = [
            {"name": "example1", "type": "dir"},
            {"name": "example2", "type": "dir"},
            {"name": "file1", "type": "file"},  # Should be filtered out
        ]
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        # Act
        result = _get_all_train_init_example_options()

        # Assert
        mock_get.assert_called_once_with(
            "https://api.github.com/repos/basetenlabs/ml-cookbook/contents/examples",
            headers={},
        )
        assert len(result) == 2
        assert "example1" in result
        assert "example2" in result
        assert "file1" not in result  # Files should be filtered out

    @patch("requests.get")
    def test_successful_request_with_token(self, mock_get):
        """Test successful API call with authentication token"""
        # Arrange
        mock_response = Mock()
        mock_response.json.return_value = [{"name": "example1", "type": "dir"}]
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        # Act
        result = _get_all_train_init_example_options(token="test_token")

        # Assert
        mock_get.assert_called_once_with(
            "https://api.github.com/repos/basetenlabs/ml-cookbook/contents/examples",
            headers={"Authorization": "token test_token"},
        )
        assert len(result) == 1
        assert "example1" in result

    @patch("requests.get")
    def test_custom_repo_and_subdir(self, mock_get):
        """Test with custom repository and subdirectory"""
        # Arrange
        mock_response = Mock()
        mock_response.json.return_value = []
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        # Act
        _ = _get_all_train_init_example_options(
            repo_id="custom-repo", examples_subdir="custom-examples"
        )

        # Assert
        mock_get.assert_called_once_with(
            "https://api.github.com/repos/basetenlabs/custom-repo/contents/custom-examples",
            headers={},
        )

    @patch("requests.get")
    def test_single_item_response(self, mock_get):
        """Test when API returns a single item instead of a list"""
        # Arrange
        mock_response = Mock()
        mock_response.json.return_value = {"name": "single_example", "type": "dir"}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        # Act
        result = _get_all_train_init_example_options()

        # Assert
        assert len(result) == 1
        assert "single_example" in result

    @patch("requests.get")
    @patch("click.echo")
    def test_request_exception_handling(self, mock_echo, mock_get):
        """Test handling of request exceptions"""
        # Arrange
        mock_get.side_effect = requests.exceptions.RequestException("Network error")

        # Act
        result = _get_all_train_init_example_options()

        # Assert
        mock_echo.assert_called_once_with(
            "Error exploring directory: Network error. Please file an issue at https://github.com/basetenlabs/truss/issues"
        )
        assert result == []

    @patch("requests.get")
    @patch("click.echo")
    def test_http_error_handling(self, mock_echo, mock_get):
        """Test handling of HTTP errors"""
        # Arrange
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            "404 Not Found"
        )
        mock_get.return_value = mock_response

        # Act
        result = _get_all_train_init_example_options()

        # Assert
        mock_echo.assert_called_once_with(
            "Error exploring directory: 404 Not Found. Please file an issue at https://github.com/basetenlabs/truss/issues"
        )
        assert result == []

    @patch("requests.get")
    def test_filters_only_directories(self, mock_get):
        """Test that only directories are returned, files are filtered out"""
        # Arrange
        mock_response = Mock()
        mock_response.json.return_value = [
            {"name": "example1", "type": "dir"},
            {"name": "readme.md", "type": "file"},
            {"name": "example2", "type": "dir"},
            {"name": "config.json", "type": "file"},
        ]
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        # Act
        result = _get_all_train_init_example_options()

        # Assert
        assert len(result) == 2
        assert "example1" in result
        assert "example2" in result
        assert "readme.md" not in result
        assert "config.json" not in result


class TestGetTrainInitExampleInfo:
    """Test cases for _get_train_init_example_info function"""

    @patch("requests.get")
    def test_request_without_token(self, mock_get):
        """Test successful API call without authentication token"""
        # Arrange
        mock_response = Mock()
        mock_response.json.return_value = [
            {"name": "file1.py", "type": "file"},
            {"name": "file2.py", "type": "file"},
        ]
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        # Act
        result = _get_train_init_example_info(example_name="test_example")

        # Assert
        mock_get.assert_called_once_with(
            "https://api.github.com/repos/basetenlabs/ml-cookbook/contents/examples/test_example",
            headers={},
        )
        assert len(result) == 0  # No training subdir in mock response

    @patch("requests.get")
    def test_successful_request_without_token(self, mock_get):
        """Test successful API call without authentication token"""
        # Arrange
        mock_response = Mock()
        mock_response.json.return_value = [
            {"name": "training", "path": "git_path_1", "type": "dir"},
            {"name": "file2.py", "type": "file"},
        ]
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        # Act
        result = _get_train_init_example_info(example_name="test_example")

        # Assert
        mock_get.assert_called_once_with(
            "https://api.github.com/repos/basetenlabs/ml-cookbook/contents/examples/test_example",
            headers={},
        )
        assert len(result) == 1  # One training subdir in mock response

    @patch("requests.get")
    def test_request_with_token(self, mock_get):
        """Test successful API call with authentication token"""
        # Arrange
        mock_response = Mock()
        mock_response.json.return_value = [{"name": "file1.py", "type": "file"}]
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        # Act
        result = _get_train_init_example_info(
            example_name="test_example", token="test_token"
        )

        # Assert
        mock_get.assert_called_once_with(
            "https://api.github.com/repos/basetenlabs/ml-cookbook/contents/examples/test_example",
            headers={"Authorization": "token test_token"},
        )
        assert len(result) == 0  # No training subdir in mock response

    @patch("requests.get")
    def test_custom_repo_and_subdir(self, mock_get):
        """Test with custom repository and subdirectory"""
        # Arrange
        mock_response = Mock()
        mock_response.json.return_value = []
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        # Act
        _ = _get_train_init_example_info(
            repo_id="custom-repo",
            examples_subdir="custom-examples",
            example_name="test_example",
        )

        # Assert
        mock_get.assert_called_once_with(
            "https://api.github.com/repos/basetenlabs/custom-repo/contents/custom-examples/test_example",
            headers={},
        )

    @patch("requests.get")
    @patch("click.echo")
    def test_404_error_returns_empty_list(self, mock_echo, mock_get):
        """Test that 404 errors return empty list without error message"""
        # Arrange
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            "404 Not Found"
        )
        mock_get.return_value = mock_response

        # Act
        result = _get_train_init_example_info(example_name="nonexistent_example")

        # Assert
        mock_echo.assert_not_called()  # Should not echo error for 404
        assert result == []

    @patch("requests.get")
    @patch("click.echo")
    def test_other_http_error_handling(self, mock_echo, mock_get):
        """Test handling of non-404 HTTP errors"""
        # Arrange
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            "500 Internal Server Error"
        )
        mock_get.return_value = mock_response

        # Act
        result = _get_train_init_example_info(example_name="test_example")

        # Assert
        mock_echo.assert_called_once_with(
            "Error exploring directory: 500 Internal Server Error. Please file an issue at https://github.com/basetenlabs/truss/issues"
        )
        assert result == []

    @patch("requests.get")
    @patch("click.echo")
    def test_request_exception_handling(self, mock_echo, mock_get):
        """Test handling of request exceptions"""
        # Arrange
        mock_get.side_effect = requests.exceptions.RequestException("Network error")

        # Act
        result = _get_train_init_example_info(example_name="test_example")

        # Assert
        mock_echo.assert_called_once_with(
            "Error exploring directory: Network error. Please file an issue at https://github.com/basetenlabs/truss/issues"
        )
        assert result == []

    @patch("requests.get")
    def test_none_example_name(self, mock_get):
        """Test with None as example_name"""
        # Arrange
        mock_response = Mock()
        mock_response.json.return_value = []
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        # Act
        result = _get_train_init_example_info(example_name=None)

        # Assert
        mock_get.assert_called_once_with(
            "https://api.github.com/repos/basetenlabs/ml-cookbook/contents/examples/None",
            headers={},
        )
        assert result == []


class TestDownloadGitDirectory:
    """Test cases for download_git_directory function"""

    @patch("os.makedirs")
    @patch("requests.get")
    @patch("builtins.open", new_callable=mock_open)
    @patch("builtins.print")
    def test_download_files_without_training_dir(
        self, mock_print, mock_file, mock_get, mock_makedirs
    ):
        """Test downloading files without a training directory"""
        # Arrange
        mock_response = Mock()
        mock_response.json.return_value = [
            {
                "name": "file1.txt",
                "type": "file",
                "download_url": "https://example.com/file1.txt",
            },
            {
                "name": "file2.py",
                "type": "file",
                "download_url": "https://example.com/file2.py",
            },
        ]
        mock_response.raise_for_status.return_value = None

        # Mock file download responses
        file_response1 = Mock()
        file_response1.content = b"file1 content"
        file_response1.raise_for_status.return_value = None

        file_response2 = Mock()
        file_response2.content = b"file2 content"
        file_response2.raise_for_status.return_value = None

        mock_get.side_effect = [mock_response, file_response1, file_response2]

        # Act
        result = download_git_directory("https://api.github.com/test", "/local/dir")

        # Assert
        assert result is True
        mock_makedirs.assert_called_once_with("/local/dir", exist_ok=True)
        assert mock_get.call_count == 3
        assert mock_file.call_count == 2

    @patch("os.makedirs")
    @patch("requests.get")
    def test_download_with_training_directory(self, mock_get, mock_makedirs):
        """Test downloading when training directory is present"""
        # Arrange
        initial_response = Mock()
        initial_response.json.return_value = [
            {
                "name": "training",
                "type": "dir",
                "url": "https://api.github.com/training",
            },
            {
                "name": "other_file.txt",
                "type": "file",
                "download_url": "https://example.com/other_file.txt",
            },
        ]
        initial_response.raise_for_status.return_value = None

        training_response = Mock()
        training_response.json.return_value = []
        training_response.raise_for_status.return_value = None

        mock_get.side_effect = [initial_response, training_response]

        # Act
        result = download_git_directory("https://api.github.com/test", "/local/dir")

        # Assert
        assert result is True
        # Should be called twice: once for initial dir, once for training contents
        assert mock_makedirs.call_count == 2

    @patch("os.makedirs")
    @patch("requests.get")
    def test_download_subdirectory_recursively(self, mock_get, mock_makedirs):
        """Test recursive download of subdirectories"""
        # Arrange
        initial_response = Mock()
        initial_response.json.return_value = [
            {"name": "subdir", "type": "dir", "url": "https://api.github.com/subdir"}
        ]
        initial_response.raise_for_status.return_value = None

        subdir_response = Mock()
        subdir_response.json.return_value = []
        subdir_response.raise_for_status.return_value = None

        mock_get.side_effect = [initial_response, subdir_response]

        # Act
        result = download_git_directory("https://api.github.com/test", "/local/dir")

        # Assert
        assert result is True
        expected_calls = [
            call("/local/dir", exist_ok=True),
            call("/local/dir/subdir", exist_ok=True),
        ]
        mock_makedirs.assert_has_calls(expected_calls)

    @patch("os.makedirs")
    @patch("requests.get")
    @patch("builtins.print")
    def test_download_with_authentication_token(
        self, mock_print, mock_get, mock_makedirs
    ):
        """Test download with authentication token"""
        # Arrange
        mock_response = Mock()
        mock_response.json.return_value = []
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        # Act
        result = download_git_directory(
            "https://api.github.com/test", "/local/dir", token="test_token"
        )

        # Assert
        assert result is True
        mock_get.assert_called_once_with(
            "https://api.github.com/test", headers={"Authorization": "token test_token"}
        )

    @patch("os.makedirs")
    @patch("requests.get")
    @patch("builtins.print")
    def test_download_single_file_response(self, mock_print, mock_get, mock_makedirs):
        """Test when API returns a single file instead of a list"""
        # Arrange
        mock_response = Mock()
        mock_response.json.return_value = {
            "name": "single_file.txt",
            "type": "file",
            "download_url": "https://example.com/single_file.txt",
        }
        mock_response.raise_for_status.return_value = None

        file_response = Mock()
        file_response.content = b"single file content"
        file_response.raise_for_status.return_value = None

        mock_get.side_effect = [mock_response, file_response]

        with patch("builtins.open", mock_open()) as mock_file:
            # Act
            result = download_git_directory("https://api.github.com/test", "/local/dir")

        # Assert
        assert result is True
        mock_file.assert_called_once_with("/local/dir/single_file.txt", "wb")

    @patch("os.makedirs")
    @patch("requests.get")
    @patch("builtins.print")
    def test_download_exception_handling(self, mock_print, mock_get, mock_makedirs):
        """Test exception handling during download"""
        # Arrange
        mock_get.side_effect = Exception("Network error")

        # Act
        result = download_git_directory("https://api.github.com/test", "/local/dir")

        # Assert
        assert result is False
        mock_print.assert_called_with("Error processing response: Network error")


if __name__ == "__main__":
    pytest.main([__file__])
