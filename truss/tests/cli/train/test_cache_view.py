from unittest.mock import Mock

import click
import pytest

from truss.cli.train.core import (
    SORT_BY_FILEPATH,
    SORT_BY_MODIFIED,
    SORT_BY_PERMISSIONS,
    SORT_BY_SIZE,
    SORT_BY_TYPE,
    SORT_ORDER_ASC,
    SORT_ORDER_DESC,
    view_cache_summary,
    view_cache_summary_by_project,
)
from truss.remote.baseten.remote import BasetenRemote


def test_view_cache_summary_success(capsys):
    """Test successful cache summary viewing."""
    mock_api = Mock()
    mock_remote = Mock(spec=BasetenRemote)
    mock_remote.api = mock_api

    mock_api.get_cache_summary.return_value = {
        "timestamp": "2024-01-01T12:00:00Z",
        "project_id": "proj123",
        "file_summaries": [
            {
                "path": "model/weights.bin",
                "size_bytes": 1024 * 1024 * 100,
                "modified": "2024-01-01T10:00:00Z",
                "file_type": "file",
                "permissions": "-rw-r--r--",
            },
            {
                "path": "config.json",
                "size_bytes": 1024,
                "modified": "2024-01-01T09:00:00Z",
                "file_type": "file",
                "permissions": "-rw-r--r--",
            },
            {
                "path": "model/",
                "size_bytes": 0,
                "modified": "2024-01-01T08:00:00Z",
                "file_type": "directory",
                "permissions": "drwxr-xr-x",
            },
        ],
    }
    mock_api.list_training_projects.return_value = [
        {"id": "proj123", "name": "test-project"}
    ]

    view_cache_summary(mock_remote, "proj123", SORT_BY_FILEPATH, SORT_ORDER_ASC)

    mock_api.get_cache_summary.assert_called_once_with("proj123")

    captured = capsys.readouterr()
    assert "Cache summary for project: proj123" in captured.out
    assert "model/weights.bin" in captured.out
    assert "config.json" in captured.out
    assert "104.86 MB" in captured.out
    assert "1.02 KB" in captured.out


def test_view_cache_summary_no_cache(capsys):
    """Test when no cache summary is found."""
    mock_api = Mock()
    mock_remote = Mock(spec=BasetenRemote)
    mock_remote.api = mock_api

    mock_api.get_cache_summary.return_value = {}

    mock_api.list_training_projects.return_value = [
        {"id": "proj123", "name": "test-project"}
    ]

    view_cache_summary(mock_remote, "proj123", SORT_BY_FILEPATH, SORT_ORDER_ASC)

    mock_api.get_cache_summary.assert_called_once_with("proj123")

    captured = capsys.readouterr()
    assert "No cache summary found for this project." in captured.out


def test_view_cache_summary_empty_files(capsys):
    """Test when cache summary exists but has no files."""
    mock_api = Mock()
    mock_remote = Mock(spec=BasetenRemote)
    mock_remote.api = mock_api

    mock_api.list_training_projects.return_value = [
        {"id": "proj123", "name": "test-project"}
    ]
    mock_api.get_cache_summary.return_value = {
        "timestamp": "2024-01-01T12:00:00Z",
        "project_id": "proj123",
        "file_summaries": [],
    }

    view_cache_summary(mock_remote, "proj123", SORT_BY_FILEPATH, SORT_ORDER_ASC)

    mock_api.get_cache_summary.assert_called_once_with("proj123")

    captured = capsys.readouterr()
    assert "No files found in cache." in captured.out


def test_view_cache_summary_api_error(capsys):
    """Test when API call fails."""
    mock_api = Mock()
    mock_remote = Mock(spec=BasetenRemote)
    mock_remote.api = mock_api

    mock_api.list_training_projects.return_value = [
        {"id": "proj123", "name": "test-project"}
    ]
    mock_api.get_cache_summary.side_effect = Exception("API Error")

    with pytest.raises(Exception, match="API Error"):
        view_cache_summary(mock_remote, "proj123", SORT_BY_FILEPATH, SORT_ORDER_ASC)

    mock_api.get_cache_summary.assert_called_once_with("proj123")

    captured = capsys.readouterr()
    assert "Error fetching cache summary: API Error" in captured.out


def test_view_cache_summary_sort_by_size_asc(capsys):
    """Test sorting by size in ascending order."""
    mock_api = Mock()
    mock_remote = Mock(spec=BasetenRemote)
    mock_remote.api = mock_api

    mock_api.list_training_projects.return_value = [
        {"id": "proj123", "name": "test-project"}
    ]
    mock_api.get_cache_summary.return_value = {
        "timestamp": "2024-01-01T12:00:00Z",
        "project_id": "proj123",
        "file_summaries": [
            {
                "path": "large_file.bin",
                "size_bytes": 1024 * 1024 * 100,
                "modified": "2024-01-01T10:00:00Z",
                "file_type": "file",
                "permissions": "-rw-r--r--",
            },
            {
                "path": "small_file.txt",
                "size_bytes": 1024,
                "modified": "2024-01-01T09:00:00Z",
                "file_type": "file",
                "permissions": "-rw-r--r--",
            },
            {
                "path": "medium_file.dat",
                "size_bytes": 1024 * 1024,
                "modified": "2024-01-01T11:00:00Z",
                "file_type": "file",
                "permissions": "-rw-r--r--",
            },
        ],
    }

    view_cache_summary(mock_remote, "proj123", SORT_BY_SIZE, SORT_ORDER_ASC)

    mock_api.get_cache_summary.assert_called_once_with("proj123")

    captured = capsys.readouterr()
    output_lines = captured.out.split("\n")

    table_start = -1
    for i, line in enumerate(output_lines):
        if "small_file.txt" in line:
            table_start = i
            break

    small_file_line = None
    medium_file_line = None
    large_file_line = None

    for line in output_lines[table_start:]:
        if "small_file.txt" in line:
            small_file_line = line
        elif "medium_file.dat" in line:
            medium_file_line = line
        elif "large_file.bin" in line:
            large_file_line = line

    assert small_file_line is not None
    assert medium_file_line is not None
    assert large_file_line is not None

    small_pos = captured.out.find("small_file.txt")
    medium_pos = captured.out.find("medium_file.dat")
    large_pos = captured.out.find("large_file.bin")

    assert small_pos < medium_pos < large_pos


def test_view_cache_summary_sort_by_size_desc(capsys):
    """Test sorting by size in descending order."""
    mock_api = Mock()
    mock_remote = Mock(spec=BasetenRemote)
    mock_remote.api = mock_api

    mock_api.list_training_projects.return_value = [
        {"id": "proj123", "name": "test-project"}
    ]
    mock_api.get_cache_summary.return_value = {
        "timestamp": "2024-01-01T12:00:00Z",
        "project_id": "proj123",
        "file_summaries": [
            {
                "path": "large_file.bin",
                "size_bytes": 1024 * 1024 * 100,
                "modified": "2024-01-01T10:00:00Z",
                "file_type": "file",
                "permissions": "-rw-r--r--",
            },
            {
                "path": "small_file.txt",
                "size_bytes": 1024,
                "modified": "2024-01-01T09:00:00Z",
                "file_type": "file",
                "permissions": "-rw-r--r--",
            },
            {
                "path": "medium_file.dat",
                "size_bytes": 1024 * 1024,
                "modified": "2024-01-01T11:00:00Z",
                "file_type": "file",
                "permissions": "-rw-r--r--",
            },
        ],
    }

    view_cache_summary(mock_remote, "proj123", SORT_BY_SIZE, SORT_ORDER_DESC)

    mock_api.get_cache_summary.assert_called_once_with("proj123")

    captured = capsys.readouterr()

    small_pos = captured.out.find("small_file.txt")
    medium_pos = captured.out.find("medium_file.dat")
    large_pos = captured.out.find("large_file.bin")

    assert large_pos < medium_pos < small_pos


def test_view_cache_summary_sort_by_modified_asc(capsys):
    """Test sorting by modified date in ascending order."""
    mock_api = Mock()
    mock_remote = Mock(spec=BasetenRemote)
    mock_remote.api = mock_api

    mock_api.list_training_projects.return_value = [
        {"id": "proj123", "name": "test-project"}
    ]
    mock_api.get_cache_summary.return_value = {
        "timestamp": "2024-01-01T12:00:00Z",
        "project_id": "proj123",
        "file_summaries": [
            {
                "path": "old_file.txt",
                "size_bytes": 1024,
                "modified": "2024-01-01T08:00:00Z",
                "file_type": "file",
                "permissions": "-rw-r--r--",
            },
            {
                "path": "new_file.txt",
                "size_bytes": 1024,
                "modified": "2024-01-01T12:00:00Z",
                "file_type": "file",
                "permissions": "-rw-r--r--",
            },
            {
                "path": "middle_file.txt",
                "size_bytes": 1024,
                "modified": "2024-01-01T10:00:00Z",
                "file_type": "file",
                "permissions": "-rw-r--r--",
            },
        ],
    }

    view_cache_summary(mock_remote, "proj123", SORT_BY_MODIFIED, SORT_ORDER_ASC)

    mock_api.get_cache_summary.assert_called_once_with("proj123")

    captured = capsys.readouterr()

    old_pos = captured.out.find("old_file.txt")
    middle_pos = captured.out.find("middle_file.txt")
    new_pos = captured.out.find("new_file.txt")

    assert old_pos < middle_pos < new_pos


def test_view_cache_summary_sort_by_filepath_desc(capsys):
    """Test sorting by filepath in descending order."""
    mock_api = Mock()
    mock_remote = Mock(spec=BasetenRemote)
    mock_remote.api = mock_api

    mock_api.get_cache_summary.return_value = {
        "timestamp": "2024-01-01T12:00:00Z",
        "project_id": "proj123",
        "file_summaries": [
            {
                "path": "a_file.txt",
                "size_bytes": 1024,
                "modified": "2024-01-01T10:00:00Z",
                "file_type": "file",
                "permissions": "-rw-r--r--",
            },
            {
                "path": "z_file.txt",
                "size_bytes": 1024,
                "modified": "2024-01-01T10:00:00Z",
                "file_type": "file",
                "permissions": "-rw-r--r--",
            },
            {
                "path": "m_file.txt",
                "size_bytes": 1024,
                "modified": "2024-01-01T10:00:00Z",
                "file_type": "file",
                "permissions": "-rw-r--r--",
            },
        ],
    }

    view_cache_summary(mock_remote, "proj123", SORT_BY_FILEPATH, SORT_ORDER_DESC)

    mock_api.get_cache_summary.assert_called_once_with("proj123")

    captured = capsys.readouterr()

    a_pos = captured.out.find("a_file.txt")
    m_pos = captured.out.find("m_file.txt")
    z_pos = captured.out.find("z_file.txt")

    assert z_pos < m_pos < a_pos


def test_view_cache_summary_by_project_name_success(capsys):
    """Test successful cache summary viewing by project name."""
    mock_api = Mock()
    mock_remote = Mock(spec=BasetenRemote)
    mock_remote.api = mock_api

    # Mock the get_cache_summary response for successful project ID lookup
    mock_api.get_cache_summary.return_value = {
        "timestamp": "2024-01-01T12:00:00Z",
        "project_id": "proj123",
        "file_summaries": [
            {
                "path": "model/weights.bin",
                "size_bytes": 1024 * 1024 * 100,
                "modified": "2024-01-01T10:00:00Z",
                "file_type": "file",
                "permissions": "-rw-r--r--",
            }
        ],
    }

    # Mock the list_training_projects response
    mock_api.list_training_projects.return_value = [
        {"id": "proj123", "name": "test-project"},
        {"id": "proj456", "name": "another-project"},
    ]

    view_cache_summary_by_project(
        mock_remote, "test-project", SORT_BY_FILEPATH, SORT_ORDER_ASC
    )

    assert mock_api.get_cache_summary.call_count == 1
    assert mock_api.list_training_projects.call_count == 1

    captured = capsys.readouterr()
    assert "Cache summary for project: proj123" in captured.out
    assert "model/weights.bin" in captured.out


def test_view_cache_summary_by_project_name_not_found(capsys):
    """Test when project name is not found."""
    mock_api = Mock()
    mock_remote = Mock(spec=BasetenRemote)
    mock_remote.api = mock_api

    mock_api.list_training_projects.return_value = [
        {"id": "proj123", "name": "test-project"},
        {"id": "proj456", "name": "another-project"},
    ]

    with pytest.raises(
        click.ClickException, match="Project 'nonexistent-project' not found"
    ):
        view_cache_summary_by_project(
            mock_remote, "nonexistent-project", SORT_BY_FILEPATH, SORT_ORDER_ASC
        )

    mock_api.list_training_projects.assert_called_once()


def test_view_cache_summary_by_project_id_direct(capsys):
    """Test that project ID is used directly."""
    mock_api = Mock()
    mock_remote = Mock(spec=BasetenRemote)
    mock_remote.api = mock_api

    mock_api.get_cache_summary.return_value = {
        "timestamp": "2024-01-01T12:00:00Z",
        "project_id": "proj123",
        "file_summaries": [
            {
                "path": "model/weights.bin",
                "size_bytes": 1024 * 1024 * 100,
                "modified": "2024-01-01T10:00:00Z",
                "file_type": "file",
                "permissions": "-rw-r--r--",
            }
        ],
    }

    mock_api.list_training_projects.return_value = [
        {"id": "proj123", "name": "test-project"},
        {"id": "proj456", "name": "another-project"},
    ]

    project_id = "proj123"
    view_cache_summary_by_project(
        mock_remote, project_id, SORT_BY_FILEPATH, SORT_ORDER_ASC
    )

    assert mock_api.get_cache_summary.call_count == 1
    assert mock_api.list_training_projects.call_count == 1

    captured = capsys.readouterr()
    assert "Cache summary for project: proj123" in captured.out


def test_view_cache_summary_by_project_other_error():
    """Test that other errors (not 404) are re-raised."""
    mock_api = Mock()
    mock_remote = Mock(spec=BasetenRemote)
    mock_remote.api = mock_api

    mock_api.list_training_projects.return_value = [
        {"id": "proj123", "name": "some-project"}
    ]
    mock_api.get_cache_summary.side_effect = Exception("Network error")

    with pytest.raises(Exception, match="Network error"):
        view_cache_summary_by_project(
            mock_remote, "some-project", SORT_BY_FILEPATH, SORT_ORDER_ASC
        )

    mock_api.get_cache_summary.assert_called_once_with("proj123")


def test_view_cache_summary_by_project_list_error(capsys):
    """Test when listing projects fails after 404."""
    mock_api = Mock()
    mock_remote = Mock(spec=BasetenRemote)
    mock_remote.api = mock_api

    mock_api.list_training_projects.side_effect = Exception("API error")

    with pytest.raises(click.ClickException, match="Error fetching project: API error"):
        view_cache_summary_by_project(
            mock_remote, "nonexistent-project", SORT_BY_FILEPATH, SORT_ORDER_ASC
        )

    mock_api.list_training_projects.assert_called_once()


def test_view_cache_summary_sort_by_type_asc(capsys):
    """Test sorting by file type in ascending order."""
    mock_api = Mock()
    mock_remote = Mock(spec=BasetenRemote)
    mock_remote.api = mock_api

    mock_api.list_training_projects.return_value = [
        {"id": "proj123", "name": "test-project"}
    ]
    mock_api.get_cache_summary.return_value = {
        "timestamp": "2024-01-01T12:00:00Z",
        "project_id": "proj123",
        "file_summaries": [
            {
                "path": "config.json",
                "size_bytes": 1024,
                "modified": "2024-01-01T09:00:00Z",
                "file_type": "file",
                "permissions": "-rw-r--r--",
            },
            {
                "path": "model/",
                "size_bytes": 0,
                "modified": "2024-01-01T08:00:00Z",
                "file_type": "directory",
                "permissions": "drwxr-xr-x",
            },
            {
                "path": "data.txt",
                "size_bytes": 2048,
                "modified": "2024-01-01T10:00:00Z",
                "file_type": "file",
                "permissions": "-rw-r--r--",
            },
        ],
    }

    view_cache_summary(mock_remote, "proj123", SORT_BY_TYPE, SORT_ORDER_ASC)

    mock_api.get_cache_summary.assert_called_once_with("proj123")

    captured = capsys.readouterr()
    output_lines = captured.out.split("\n")

    table_start = -1
    for i, line in enumerate(output_lines):
        if "config.json" in line or "data.txt" in line or "model/" in line:
            table_start = i
            break

    directory_line = None
    file_lines = []

    for line in output_lines[table_start:]:
        if "model/" in line:
            directory_line = line
        elif "config.json" in line or "data.txt" in line:
            file_lines.append(line)

    assert directory_line is not None
    assert len(file_lines) == 2

    directory_pos = captured.out.find("model/")
    config_pos = captured.out.find("config.json")
    data_pos = captured.out.find("data.txt")

    assert directory_pos < config_pos
    assert directory_pos < data_pos


def test_view_cache_summary_sort_by_type_desc(capsys):
    """Test sorting by file type in descending order."""
    mock_api = Mock()
    mock_remote = Mock(spec=BasetenRemote)
    mock_remote.api = mock_api

    mock_api.list_training_projects.return_value = [
        {"id": "proj123", "name": "test-project"}
    ]
    mock_api.get_cache_summary.return_value = {
        "timestamp": "2024-01-01T12:00:00Z",
        "project_id": "proj123",
        "file_summaries": [
            {
                "path": "config.json",
                "size_bytes": 1024,
                "modified": "2024-01-01T09:00:00Z",
                "file_type": "file",
                "permissions": "-rw-r--r--",
            },
            {
                "path": "model/",
                "size_bytes": 0,
                "modified": "2024-01-01T08:00:00Z",
                "file_type": "directory",
                "permissions": "drwxr-xr-x",
            },
            {
                "path": "data.txt",
                "size_bytes": 2048,
                "modified": "2024-01-01T10:00:00Z",
                "file_type": "file",
                "permissions": "-rw-r--r--",
            },
        ],
    }

    view_cache_summary(mock_remote, "proj123", SORT_BY_TYPE, SORT_ORDER_DESC)

    mock_api.get_cache_summary.assert_called_once_with("proj123")

    captured = capsys.readouterr()

    config_pos = captured.out.find("config.json")
    data_pos = captured.out.find("data.txt")
    directory_pos = captured.out.find("model/")

    assert config_pos < directory_pos
    assert data_pos < directory_pos


def test_view_cache_summary_sort_by_permissions_asc(capsys):
    """Test sorting by permissions in ascending order."""
    mock_api = Mock()
    mock_remote = Mock(spec=BasetenRemote)
    mock_remote.api = mock_api

    mock_api.list_training_projects.return_value = [
        {"id": "proj123", "name": "test-project"}
    ]
    mock_api.get_cache_summary.return_value = {
        "timestamp": "2024-01-01T12:00:00Z",
        "project_id": "proj123",
        "file_summaries": [
            {
                "path": "config.json",
                "size_bytes": 1024,
                "modified": "2024-01-01T09:00:00Z",
                "file_type": "file",
                "permissions": "-rw-r--r--",
            },
            {
                "path": "model/",
                "size_bytes": 0,
                "modified": "2024-01-01T08:00:00Z",
                "file_type": "directory",
                "permissions": "drwxr-xr-x",
            },
            {
                "path": "script.sh",
                "size_bytes": 2048,
                "modified": "2024-01-01T10:00:00Z",
                "file_type": "file",
                "permissions": "-rwxr-xr-x",
            },
        ],
    }

    view_cache_summary(mock_remote, "proj123", SORT_BY_PERMISSIONS, SORT_ORDER_ASC)

    mock_api.get_cache_summary.assert_called_once_with("proj123")

    captured = capsys.readouterr()

    config_pos = captured.out.find("config.json")
    script_pos = captured.out.find("script.sh")
    directory_pos = captured.out.find("model/")

    assert config_pos != -1, "Config file not found in output"
    assert script_pos != -1, "Script file not found in output"
    assert directory_pos != -1, "Directory not found in output"

    assert config_pos < script_pos
    assert script_pos < directory_pos


def test_view_cache_summary_sort_by_permissions_desc(capsys):
    """Test sorting by permissions in descending order."""
    mock_api = Mock()
    mock_remote = Mock(spec=BasetenRemote)
    mock_remote.api = mock_api

    mock_api.list_training_projects.return_value = [
        {"id": "proj123", "name": "test-project"}
    ]
    mock_api.get_cache_summary.return_value = {
        "timestamp": "2024-01-01T12:00:00Z",
        "project_id": "proj123",
        "file_summaries": [
            {
                "path": "config.json",
                "size_bytes": 1024,
                "modified": "2024-01-01T09:00:00Z",
                "file_type": "file",
                "permissions": "-rw-r--r--",
            },
            {
                "path": "model/",
                "size_bytes": 0,
                "modified": "2024-01-01T08:00:00Z",
                "file_type": "directory",
                "permissions": "drwxr-xr-x",
            },
            {
                "path": "script.sh",
                "size_bytes": 2048,
                "modified": "2024-01-01T10:00:00Z",
                "file_type": "file",
                "permissions": "-rwxr-xr-x",
            },
        ],
    }

    view_cache_summary(mock_remote, "proj123", SORT_BY_PERMISSIONS, SORT_ORDER_DESC)

    mock_api.get_cache_summary.assert_called_once_with("proj123")

    captured = capsys.readouterr()

    directory_pos = captured.out.find("model/")
    script_pos = captured.out.find("script.sh")
    config_pos = captured.out.find("config.json")

    assert directory_pos != -1, "Directory not found in output"
    assert script_pos != -1, "Script file not found in output"
    assert config_pos != -1, "Config file not found in output"

    assert directory_pos < script_pos
    assert script_pos < config_pos
