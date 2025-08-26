from unittest.mock import Mock

import pytest

from truss.cli.train.core import (
    SORT_BY_FILEPATH,
    SORT_BY_MODIFIED,
    SORT_BY_SIZE,
    SORT_ORDER_ASC,
    SORT_ORDER_DESC,
    view_cache_summary,
)
from truss.remote.baseten.remote import BasetenRemote


def test_view_cache_summary_success(capsys):
    """Test successful cache structure viewing."""
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
            },
            {
                "path": "config.json",
                "size_bytes": 1024,
                "modified": "2024-01-01T09:00:00Z",
            },
        ],
    }

    view_cache_summary(mock_remote, "proj123", SORT_BY_FILEPATH, SORT_ORDER_ASC)

    mock_api.get_cache_summary.assert_called_once_with("proj123")

    captured = capsys.readouterr()
    assert "Cache Structure for Project: proj123" in captured.out
    assert "model/weights.bin" in captured.out
    assert "config.json" in captured.out
    assert "104.86 MB" in captured.out
    assert "1.02 KB" in captured.out
    assert "Total files: 2" in captured.out
    assert "Total size: 104.86 MB" in captured.out


def test_view_cache_summary_no_cache(capsys):
    """Test when no cache structure is found."""
    mock_api = Mock()
    mock_remote = Mock(spec=BasetenRemote)
    mock_remote.api = mock_api

    mock_api.get_cache_summary.return_value = {}

    view_cache_summary(mock_remote, "proj123", SORT_BY_FILEPATH, SORT_ORDER_ASC)

    mock_api.get_cache_summary.assert_called_once_with("proj123")

    captured = capsys.readouterr()
    assert "No cache structure found for this project." in captured.out


def test_view_cache_summary_empty_files(capsys):
    """Test when cache structure exists but has no files."""
    mock_api = Mock()
    mock_remote = Mock(spec=BasetenRemote)
    mock_remote.api = mock_api

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

    mock_api.get_cache_summary.side_effect = Exception("API Error")

    with pytest.raises(Exception, match="API Error"):
        view_cache_summary(mock_remote, "proj123", SORT_BY_FILEPATH, SORT_ORDER_ASC)

    mock_api.get_cache_summary.assert_called_once_with("proj123")

    captured = capsys.readouterr()
    assert "Error fetching cache structure: API Error" in captured.out


def test_view_cache_summary_sort_by_size_asc(capsys):
    """Test sorting by size in ascending order."""
    mock_api = Mock()
    mock_remote = Mock(spec=BasetenRemote)
    mock_remote.api = mock_api

    mock_api.get_cache_summary.return_value = {
        "timestamp": "2024-01-01T12:00:00Z",
        "project_id": "proj123",
        "file_summaries": [
            {
                "path": "large_file.bin",
                "size_bytes": 1024 * 1024 * 100,
                "modified": "2024-01-01T10:00:00Z",
            },
            {
                "path": "small_file.txt",
                "size_bytes": 1024,
                "modified": "2024-01-01T09:00:00Z",
            },
            {
                "path": "medium_file.dat",
                "size_bytes": 1024 * 1024,
                "modified": "2024-01-01T11:00:00Z",
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

    mock_api.get_cache_summary.return_value = {
        "timestamp": "2024-01-01T12:00:00Z",
        "project_id": "proj123",
        "file_summaries": [
            {
                "path": "large_file.bin",
                "size_bytes": 1024 * 1024 * 100,
                "modified": "2024-01-01T10:00:00Z",
            },
            {
                "path": "small_file.txt",
                "size_bytes": 1024,
                "modified": "2024-01-01T09:00:00Z",
            },
            {
                "path": "medium_file.dat",
                "size_bytes": 1024 * 1024,
                "modified": "2024-01-01T11:00:00Z",
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

    mock_api.get_cache_summary.return_value = {
        "timestamp": "2024-01-01T12:00:00Z",
        "project_id": "proj123",
        "file_summaries": [
            {
                "path": "old_file.txt",
                "size_bytes": 1024,
                "modified": "2024-01-01T09:00:00Z",
            },
            {
                "path": "new_file.txt",
                "size_bytes": 1024,
                "modified": "2024-01-01T11:00:00Z",
            },
            {
                "path": "middle_file.txt",
                "size_bytes": 1024,
                "modified": "2024-01-01T10:00:00Z",
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
            },
            {
                "path": "z_file.txt",
                "size_bytes": 1024,
                "modified": "2024-01-01T10:00:00Z",
            },
            {
                "path": "m_file.txt",
                "size_bytes": 1024,
                "modified": "2024-01-01T10:00:00Z",
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
