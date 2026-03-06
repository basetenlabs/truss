import json
from unittest.mock import Mock, patch

import pytest

from truss.cli.train.checkpoint import (
    OUTPUT_FORMAT_CSV,
    OUTPUT_FORMAT_JSON,
    SORT_BY_CREATED,
    SORT_BY_SIZE,
    SORT_ORDER_ASC,
    SORT_ORDER_DESC,
    _build_directory_listing,
    _fetch_and_display_file,
    view_checkpoint_list,
)
from truss.remote.baseten.remote import BasetenRemote

SAMPLE_CHECKPOINTS = [
    {
        "checkpoint_id": "ckpt-001",
        "checkpoint_type": "lora",
        "base_model": "meta-llama/Llama-3-8B",
        "size_bytes": 1024 * 1024 * 50,
        "created_at": "2024-06-01T10:00:00Z",
        "lora_adapter_config": {"r": 16},
    },
    {
        "checkpoint_id": "ckpt-002",
        "checkpoint_type": "full",
        "base_model": None,
        "size_bytes": 1024 * 1024 * 1024 * 2,
        "created_at": "2024-06-01T12:00:00Z",
    },
    {
        "checkpoint_id": "ckpt-003",
        "checkpoint_type": "lora",
        "base_model": "meta-llama/Llama-3-8B",
        "size_bytes": 1024 * 1024 * 100,
        "created_at": "2024-06-01T11:00:00Z",
        "lora_adapter_config": {"r": 32},
    },
]

SAMPLE_FILES = [
    {
        "url": "https://example.com/file1",
        "relative_file_name": "ckpt-001/adapter_model.safetensors",
        "node_rank": 0,
        "size_bytes": 1024 * 1024 * 50,
        "last_modified": "2024-06-01T10:00:00Z",
    },
    {
        "url": "https://example.com/file2",
        "relative_file_name": "ckpt-001/adapter_config.json",
        "node_rank": 0,
        "size_bytes": 512,
        "last_modified": "2024-06-01T10:00:00Z",
    },
    {
        "url": "https://example.com/file3",
        "relative_file_name": "ckpt-002/model.safetensors",
        "node_rank": 0,
        "size_bytes": 1024 * 1024 * 1024 * 2,
        "last_modified": "2024-06-01T12:00:00Z",
    },
]


def _make_mock_remote():
    mock_api = Mock()
    mock_remote = Mock(spec=BasetenRemote)
    mock_remote.api = mock_api
    return mock_remote, mock_api


def test_view_checkpoint_list_success(capsys):
    """Test successful checkpoint listing with 3 checkpoints."""
    mock_remote, mock_api = _make_mock_remote()
    mock_api.list_training_job_checkpoints.return_value = {
        "checkpoints": SAMPLE_CHECKPOINTS
    }

    view_checkpoint_list(
        mock_remote, "proj123", "job456", SORT_BY_CREATED, SORT_ORDER_ASC
    )

    mock_api.list_training_job_checkpoints.assert_called_once_with("proj123", "job456")

    captured = capsys.readouterr()
    assert "ckpt-001" in captured.out
    assert "ckpt-002" in captured.out
    assert "ckpt-003" in captured.out
    assert "lora" in captured.out
    assert "full" in captured.out
    assert "Checkpoints for job: job456" in captured.out


def test_view_checkpoint_list_no_checkpoints(capsys):
    """Test output when no checkpoints are found."""
    mock_remote, mock_api = _make_mock_remote()
    mock_api.list_training_job_checkpoints.return_value = {"checkpoints": []}

    view_checkpoint_list(
        mock_remote, "proj123", "job456", SORT_BY_CREATED, SORT_ORDER_ASC
    )

    captured = capsys.readouterr()
    assert "No checkpoints found for job: job456." in captured.out


def test_view_checkpoint_list_sort_by_size(capsys):
    """Test sorting checkpoints by size."""
    mock_remote, mock_api = _make_mock_remote()
    mock_api.list_training_job_checkpoints.return_value = {
        "checkpoints": SAMPLE_CHECKPOINTS
    }

    view_checkpoint_list(mock_remote, "proj123", "job456", SORT_BY_SIZE, SORT_ORDER_ASC)

    captured = capsys.readouterr()
    # ckpt-001 (50 MB) should appear before ckpt-003 (100 MB) before ckpt-002 (2 GB)
    pos_001 = captured.out.find("ckpt-001")
    pos_003 = captured.out.find("ckpt-003")
    pos_002 = captured.out.find("ckpt-002")
    assert pos_001 < pos_003 < pos_002


def test_view_checkpoint_list_sort_by_size_desc(capsys):
    """Test sorting checkpoints by size descending."""
    mock_remote, mock_api = _make_mock_remote()
    mock_api.list_training_job_checkpoints.return_value = {
        "checkpoints": SAMPLE_CHECKPOINTS
    }

    view_checkpoint_list(
        mock_remote, "proj123", "job456", SORT_BY_SIZE, SORT_ORDER_DESC
    )

    captured = capsys.readouterr()
    # ckpt-002 (2 GB) should appear before ckpt-003 (100 MB) before ckpt-001 (50 MB)
    pos_002 = captured.out.find("ckpt-002")
    pos_003 = captured.out.find("ckpt-003")
    pos_001 = captured.out.find("ckpt-001")
    assert pos_002 < pos_003 < pos_001


def test_view_checkpoint_list_csv_format(capsys):
    """Test CSV output format."""
    mock_remote, mock_api = _make_mock_remote()
    mock_api.list_training_job_checkpoints.return_value = {
        "checkpoints": SAMPLE_CHECKPOINTS
    }

    view_checkpoint_list(
        mock_remote,
        "proj123",
        "job456",
        SORT_BY_CREATED,
        SORT_ORDER_ASC,
        OUTPUT_FORMAT_CSV,
    )

    captured = capsys.readouterr()
    lines = captured.out.strip().split("\n")
    assert len(lines) == 4  # Header + 3 data rows
    assert "Checkpoint ID" in lines[0]
    assert "Type" in lines[0]
    assert "Size (bytes)" in lines[0]
    assert "ckpt-001" in lines[1]


def test_view_checkpoint_list_json_format(capsys):
    """Test JSON output format including lora_adapter_config."""
    mock_remote, mock_api = _make_mock_remote()
    mock_api.list_training_job_checkpoints.return_value = {
        "checkpoints": SAMPLE_CHECKPOINTS
    }

    view_checkpoint_list(
        mock_remote,
        "proj123",
        "job456",
        SORT_BY_CREATED,
        SORT_ORDER_ASC,
        OUTPUT_FORMAT_JSON,
    )

    captured = capsys.readouterr()
    output = json.loads(captured.out)
    assert output["job_id"] == "job456"
    assert output["total_checkpoints"] == 3
    assert len(output["checkpoints"]) == 3
    # First checkpoint (ckpt-001 after sort by created asc) should have lora_adapter_config
    ckpt_001 = next(
        c for c in output["checkpoints"] if c["checkpoint_id"] == "ckpt-001"
    )
    assert ckpt_001["lora_adapter_config"] == {"r": 16}
    # ckpt-002 (full) should not have lora_adapter_config
    ckpt_002 = next(
        c for c in output["checkpoints"] if c["checkpoint_id"] == "ckpt-002"
    )
    assert "lora_adapter_config" not in ckpt_002


def test_view_checkpoint_list_json_no_checkpoints(capsys):
    """Test JSON output when no checkpoints are found."""
    mock_remote, mock_api = _make_mock_remote()
    mock_api.list_training_job_checkpoints.return_value = {"checkpoints": []}

    view_checkpoint_list(
        mock_remote,
        "proj123",
        "job456",
        SORT_BY_CREATED,
        SORT_ORDER_ASC,
        OUTPUT_FORMAT_JSON,
    )

    captured = capsys.readouterr()
    output = json.loads(captured.out)
    assert output["job_id"] == "job456"
    assert output["total_checkpoints"] == 0
    assert output["checkpoints"] == []


def test_build_directory_listing_with_checkpoint_lookup():
    """Test that directories matching checkpoint IDs are annotated."""
    files = [
        {"_rel_path": "ckpt-001/adapter_model.safetensors", "size_bytes": 1000},
        {"_rel_path": "ckpt-001/adapter_config.json", "size_bytes": 200},
        {"_rel_path": "ckpt-002/model.safetensors", "size_bytes": 5000},
        {"_rel_path": "other-dir/data.bin", "size_bytes": 300},
    ]
    checkpoint_lookup = {
        "ckpt-001": {
            "checkpoint_type": "lora",
            "base_model": "meta-llama/Llama-3-8B",
            "size_bytes": 1024 * 1024 * 50,
        },
        "ckpt-002": {
            "checkpoint_type": "full",
            "base_model": "",
            "size_bytes": 1024 * 1024 * 1024 * 2,
        },
    }
    dirs, dir_files = _build_directory_listing(files, "", checkpoint_lookup)

    assert len(dirs) == 3
    assert len(dir_files) == 0

    dirs_by_name = {d["name"]: d for d in dirs}

    # ckpt-001 should be annotated
    assert dirs_by_name["ckpt-001"]["checkpoint_type"] == "lora"
    assert dirs_by_name["ckpt-001"]["base_model"] == "meta-llama/Llama-3-8B"
    assert dirs_by_name["ckpt-001"]["size_bytes"] == 1024 * 1024 * 50

    # ckpt-002 should be annotated
    assert dirs_by_name["ckpt-002"]["checkpoint_type"] == "full"

    # other-dir should NOT be annotated
    assert "checkpoint_type" not in dirs_by_name["other-dir"]


def test_build_directory_listing_without_checkpoint_lookup():
    """Test that directories are not annotated when no lookup is provided."""
    files = [{"_rel_path": "ckpt-001/adapter_model.safetensors", "size_bytes": 1000}]
    dirs, _ = _build_directory_listing(files, "")

    assert len(dirs) == 1
    assert "checkpoint_type" not in dirs[0]


@patch("truss.cli.train.checkpoint._open_in_pager")
@patch("truss.cli.train.checkpoint.requests.get")
def test_fetch_and_display_json_file(mock_get, mock_pager):
    """Test fetching and displaying a JSON file pretty-prints before opening pager."""
    mock_response = Mock()
    mock_response.text = '{"key": "value", "nested": {"a": 1}}'
    mock_response.raise_for_status = Mock()
    mock_get.return_value = mock_response

    file_info = {
        "url": "https://example.com/config.json",
        "relative_file_name": "ckpt-001/adapter_config.json",
        "size_bytes": 100,
    }
    _fetch_and_display_file(file_info)

    mock_get.assert_called_once_with("https://example.com/config.json", timeout=30)
    content_arg = mock_pager.call_args[0][0]
    assert '"key": "value"' in content_arg
    assert '"nested"' in content_arg


@patch("truss.cli.train.checkpoint._open_in_pager")
@patch("truss.cli.train.checkpoint.requests.get")
def test_fetch_and_display_text_file(mock_get, mock_pager):
    """Test fetching and displaying a non-JSON text file opens pager."""
    mock_response = Mock()
    mock_response.text = "some plain text content\nline 2"
    mock_response.raise_for_status = Mock()
    mock_get.return_value = mock_response

    file_info = {
        "url": "https://example.com/readme.txt",
        "relative_file_name": "ckpt-001/readme.txt",
        "size_bytes": 100,
    }
    _fetch_and_display_file(file_info)

    content_arg = mock_pager.call_args[0][0]
    assert "some plain text content" in content_arg


def test_build_directory_listing_basic():
    """Test that dirs and files are correctly separated at root level."""
    files = [
        {"_rel_path": "rank-0/weights.bin", "size_bytes": 1000},
        {"_rel_path": "rank-0/config.json", "size_bytes": 200},
        {"_rel_path": "adapter_config.json", "size_bytes": 512},
        {"_rel_path": "args.json", "size_bytes": 1024},
    ]
    dirs, dir_files = _build_directory_listing(files, "")

    assert len(dirs) == 1
    assert dirs[0]["name"] == "rank-0"
    assert len(dir_files) == 2
    file_names = {f["_rel_path"] for f in dir_files}
    assert file_names == {"adapter_config.json", "args.json"}


def test_build_directory_listing_nested():
    """Test multi-level nesting only shows immediate children."""
    files = [
        {"_rel_path": "rank-0/sub/deep.bin", "size_bytes": 500},
        {"_rel_path": "rank-0/top.json", "size_bytes": 100},
        {"_rel_path": "rank-1/data.bin", "size_bytes": 300},
    ]
    # Listing from inside rank-0
    dirs, dir_files = _build_directory_listing(files, "rank-0")

    assert len(dirs) == 1
    assert dirs[0]["name"] == "sub"
    assert dirs[0]["file_count"] == 1
    assert dirs[0]["total_size"] == 500
    assert len(dir_files) == 1
    assert dir_files[0]["_rel_path"] == "rank-0/top.json"


def test_build_directory_listing_empty():
    """Test empty current path returns root contents correctly."""
    files = [
        {"_rel_path": "file_a.txt", "size_bytes": 10},
        {"_rel_path": "file_b.txt", "size_bytes": 20},
    ]
    dirs, dir_files = _build_directory_listing(files, "")

    assert len(dirs) == 0
    assert len(dir_files) == 2


def test_build_directory_listing_aggregates_size():
    """Test that directory sizes sum correctly across all nested files."""
    files = [
        {"_rel_path": "rank-0/a.bin", "size_bytes": 1000},
        {"_rel_path": "rank-0/b.bin", "size_bytes": 2000},
        {"_rel_path": "rank-0/sub/c.bin", "size_bytes": 3000},
    ]
    dirs, dir_files = _build_directory_listing(files, "")

    assert len(dirs) == 1
    assert dirs[0]["name"] == "rank-0"
    assert dirs[0]["total_size"] == 6000
    assert dirs[0]["file_count"] == 3
    assert len(dir_files) == 0


def test_view_checkpoint_list_api_error(capsys):
    """Test error handling when API call fails."""
    mock_remote, mock_api = _make_mock_remote()
    mock_api.list_training_job_checkpoints.side_effect = Exception("API Error")

    with pytest.raises(Exception, match="API Error"):
        view_checkpoint_list(
            mock_remote, "proj123", "job456", SORT_BY_CREATED, SORT_ORDER_ASC
        )

    captured = capsys.readouterr()
    assert "Error fetching checkpoints: API Error" in captured.out
