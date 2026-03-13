import json
import struct
from unittest.mock import Mock, patch

import pytest
import requests

from truss.cli.train.checkpoint_viewer import (
    EXIT_OPTION,
    MAX_SAFETENSOR_HEADER_SIZE,
    OUTPUT_FORMAT_CSV,
    OUTPUT_FORMAT_JSON,
    SORT_BY_CREATED,
    SORT_BY_SIZE,
    SORT_ORDER_ASC,
    SORT_ORDER_DESC,
    SafetensorSummary,
    TensorSummary,
    _build_directory_listing,
    _build_explorer_choices,
    _explore_files,
    _fetch_and_display_file,
    _fetch_safetensor_header,
    _highlight_content,
    _select_checkpoint,
    _view_safetensor_file,
    view_checkpoint_list,
)
from truss.remote.baseten.remote import BasetenRemote

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

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
        "size_bytes": 1024 * 1024 * 50,
    },
    {
        "url": "https://example.com/file2",
        "relative_file_name": "ckpt-001/adapter_config.json",
        "size_bytes": 512,
    },
    {
        "url": "https://example.com/file3",
        "relative_file_name": "ckpt-002/model.safetensors",
        "size_bytes": 1024 * 1024 * 1024 * 2,
    },
]


def _make_mock_remote(checkpoints=None, files=None):
    mock_api = Mock()
    mock_remote = Mock(spec=BasetenRemote)
    mock_remote.api = mock_api
    if checkpoints is not None:
        mock_api.list_training_job_checkpoints.return_value = {
            "checkpoints": checkpoints
        }
    if files is not None:
        mock_api.get_training_job_checkpoint_presigned_url.return_value = files
    return mock_remote, mock_api


# ---------------------------------------------------------------------------
# TensorSummary
# ---------------------------------------------------------------------------


def test_tensor_summary_known_dtype():
    info = {"dtype": "F32", "shape": [768, 768]}
    t = TensorSummary.from_header_entry("layer.weight", info)
    assert t.name == "layer.weight"
    assert t.dtype == "F32"
    assert t.shape == [768, 768]
    assert t.num_params == 768 * 768
    assert t.size_bytes == 768 * 768 * 4


def test_tensor_summary_unknown_dtype():
    info = {"dtype": "UNKNOWN", "shape": [10, 10]}
    t = TensorSummary.from_header_entry("x", info)
    assert t.size_bytes is None


def test_tensor_summary_scalar():
    info = {"dtype": "I32", "shape": []}
    t = TensorSummary.from_header_entry("step", info)
    assert t.num_params == 1
    assert t.size_bytes == 4


def test_tensor_summary_bf16():
    info = {"dtype": "BF16", "shape": [1024, 4096]}
    t = TensorSummary.from_header_entry("attn.q_proj", info)
    assert t.size_bytes == 1024 * 4096 * 2


# ---------------------------------------------------------------------------
# SafetensorSummary
# ---------------------------------------------------------------------------


def test_safetensor_summary_from_header_basic():
    header = {
        "layer.weight": {"dtype": "F32", "shape": [4, 4]},
        "layer.bias": {"dtype": "F32", "shape": [4]},
    }
    summary = SafetensorSummary.from_header(header)
    assert len(summary.tensors) == 2
    assert summary.metadata == {}


def test_safetensor_summary_extracts_metadata():
    header = {
        "__metadata__": {"format": "pt", "framework": "pytorch"},
        "w": {"dtype": "F16", "shape": [2, 2]},
    }
    summary = SafetensorSummary.from_header(header)
    assert summary.metadata == {"format": "pt", "framework": "pytorch"}
    assert len(summary.tensors) == 1


def test_safetensor_summary_str_includes_summary_line():
    header = {"w": {"dtype": "F32", "shape": [2, 2]}}
    summary = SafetensorSummary.from_header(header)
    text = str(summary)
    assert "Tensors: 1" in text
    assert "Parameters: 4" in text


def test_safetensor_summary_str_unknown_dtype_shows_question_mark():
    header = {"w": {"dtype": "CUSTOM", "shape": [10]}}
    summary = SafetensorSummary.from_header(header)
    assert "?" in str(summary)


def test_safetensor_summary_str_includes_metadata():
    header = {"__metadata__": {"author": "test"}, "w": {"dtype": "F32", "shape": [1]}}
    summary = SafetensorSummary.from_header(header)
    text = str(summary)
    assert "Metadata:" in text
    assert "author: test" in text


def test_safetensor_summary_tensors_sorted_by_name():
    header = {
        "z_layer": {"dtype": "F32", "shape": [1]},
        "a_layer": {"dtype": "F32", "shape": [1]},
    }
    summary = SafetensorSummary.from_header(header)
    assert summary.tensors[0].name == "a_layer"
    assert summary.tensors[1].name == "z_layer"


# ---------------------------------------------------------------------------
# _fetch_safetensor_header
# ---------------------------------------------------------------------------


@patch("truss.cli.train.checkpoint_viewer.requests.get")
def test_fetch_safetensor_header_happy_path(mock_get):
    header_data = json.dumps({"w": {"dtype": "F32", "shape": [4]}}).encode()

    size_resp = Mock()
    size_resp.content = struct.pack("<Q", len(header_data))
    size_resp.raise_for_status = Mock()

    header_resp = Mock()
    header_resp.content = header_data
    header_resp.raise_for_status = Mock()

    mock_get.side_effect = [size_resp, header_resp]
    result = _fetch_safetensor_header("https://example.com/model.safetensors")
    assert result == {"w": {"dtype": "F32", "shape": [4]}}


@patch("truss.cli.train.checkpoint_viewer.requests.get")
def test_fetch_safetensor_header_too_large_returns_none(mock_get):
    size_resp = Mock()
    size_resp.content = struct.pack("<Q", MAX_SAFETENSOR_HEADER_SIZE + 1)
    size_resp.raise_for_status = Mock()
    mock_get.return_value = size_resp

    result = _fetch_safetensor_header("https://example.com/model.safetensors")
    assert result is None
    assert mock_get.call_count == 1


@patch("truss.cli.train.checkpoint_viewer.requests.get")
def test_fetch_safetensor_header_network_error_returns_none(mock_get):
    mock_get.side_effect = requests.RequestException("timeout")
    assert _fetch_safetensor_header("https://example.com/model.safetensors") is None


# ---------------------------------------------------------------------------
# _highlight_content
# ---------------------------------------------------------------------------


def test_highlight_content_unknown_extension_returns_original():
    content = "some raw content"
    assert _highlight_content(content, "file.unknownxyz") == content


def test_highlight_content_known_extension_returns_string():
    content = 'import os\nprint("hello")'
    result = _highlight_content(content, "script.py")
    assert isinstance(result, str)
    assert "print" in result


# ---------------------------------------------------------------------------
# _build_directory_listing
# ---------------------------------------------------------------------------


def test_build_directory_listing_with_checkpoint_lookup():
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
    assert dirs_by_name["ckpt-001"]["checkpoint_type"] == "lora"
    assert dirs_by_name["ckpt-001"]["base_model"] == "meta-llama/Llama-3-8B"
    assert dirs_by_name["ckpt-002"]["checkpoint_type"] == "full"
    assert "checkpoint_type" not in dirs_by_name["other-dir"]


def test_build_directory_listing_without_checkpoint_lookup():
    files = [{"_rel_path": "ckpt-001/adapter_model.safetensors", "size_bytes": 1000}]
    dirs, _ = _build_directory_listing(files, "")
    assert len(dirs) == 1
    assert "checkpoint_type" not in dirs[0]


def test_build_directory_listing_basic():
    files = [
        {"_rel_path": "rank-0/weights.bin", "size_bytes": 1000},
        {"_rel_path": "rank-0/config.json", "size_bytes": 200},
        {"_rel_path": "adapter_config.json", "size_bytes": 512},
        {"_rel_path": "args.json", "size_bytes": 1024},
    ]
    dirs, dir_files = _build_directory_listing(files, "")
    assert len(dirs) == 1
    assert dirs[0]["name"] == "rank-0"
    assert {f["_rel_path"] for f in dir_files} == {"adapter_config.json", "args.json"}


def test_build_directory_listing_nested():
    files = [
        {"_rel_path": "rank-0/sub/deep.bin", "size_bytes": 500},
        {"_rel_path": "rank-0/top.json", "size_bytes": 100},
        {"_rel_path": "rank-1/data.bin", "size_bytes": 300},
    ]
    dirs, dir_files = _build_directory_listing(files, "rank-0")
    assert len(dirs) == 1
    assert dirs[0]["name"] == "sub"
    assert dirs[0]["file_count"] == 1
    assert dir_files[0]["_rel_path"] == "rank-0/top.json"


def test_build_directory_listing_empty():
    files = [
        {"_rel_path": "file_a.txt", "size_bytes": 10},
        {"_rel_path": "file_b.txt", "size_bytes": 20},
    ]
    dirs, dir_files = _build_directory_listing(files, "")
    assert len(dirs) == 0
    assert len(dir_files) == 2


def test_build_directory_listing_aggregates_size():
    files = [
        {"_rel_path": "rank-0/a.bin", "size_bytes": 1000},
        {"_rel_path": "rank-0/b.bin", "size_bytes": 2000},
        {"_rel_path": "rank-0/sub/c.bin", "size_bytes": 3000},
    ]
    dirs, dir_files = _build_directory_listing(files, "")
    assert dirs[0]["total_size"] == 6000
    assert dirs[0]["file_count"] == 3
    assert len(dir_files) == 0


# ---------------------------------------------------------------------------
# _build_explorer_choices
# ---------------------------------------------------------------------------


def test_build_explorer_choices_no_parent():
    dirs = [{"name": "rank-0", "total_size": 1000, "file_count": 2}]
    files = [{"_rel_path": "config.json", "size_bytes": 100}]
    choices = _build_explorer_choices(dirs, files, has_parent=False)
    names = [c["name"] for c in choices]
    assert ".." not in names
    assert any("rank-0" in n for n in names)
    assert any("config.json" in n for n in names)
    assert choices[-1]["name"] == EXIT_OPTION


def test_build_explorer_choices_with_parent():
    choices = _build_explorer_choices([], [], has_parent=True)
    assert choices[0] == {"name": "..", "value": ("back", None)}


def test_build_explorer_choices_dir_value_is_tuple():
    dirs = [{"name": "rank-0", "total_size": 0, "file_count": 0}]
    choices = _build_explorer_choices(dirs, [], has_parent=False)
    dir_choice = next(c for c in choices if "rank-0" in c["name"])
    assert dir_choice["value"] == ("dir", "rank-0")


def test_build_explorer_choices_file_value_is_tuple():
    f = {"_rel_path": "config.json", "size_bytes": 100}
    choices = _build_explorer_choices([], [f], has_parent=False)
    file_choice = next(c for c in choices if "config.json" in c["name"])
    assert file_choice["value"] == ("file", f)


def test_build_explorer_choices_checkpoint_annotation():
    dirs = [
        {
            "name": "ckpt-001",
            "total_size": 1000,
            "file_count": 1,
            "checkpoint_type": "lora",
            "base_model": "meta-llama/Llama-3-8B",
        }
    ]
    choices = _build_explorer_choices(dirs, [], has_parent=False)
    label = next(c["name"] for c in choices if "ckpt-001" in c["name"])
    assert "lora" in label
    assert "meta-llama/Llama-3-8B" in label


def test_build_explorer_choices_dirs_sorted():
    dirs = [
        {"name": "z-dir", "total_size": 0, "file_count": 0},
        {"name": "a-dir", "total_size": 0, "file_count": 0},
    ]
    choices = _build_explorer_choices(dirs, [], has_parent=False)
    dir_names = [c["name"] for c in choices if c["value"][0] == "dir"]
    assert dir_names[0].startswith("a-dir")
    assert dir_names[1].startswith("z-dir")


# ---------------------------------------------------------------------------
# view_checkpoint_list — non-interactive
# ---------------------------------------------------------------------------


def test_view_checkpoint_list_success(capsys):
    mock_remote, mock_api = _make_mock_remote(checkpoints=SAMPLE_CHECKPOINTS)
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


def test_view_checkpoint_list_no_checkpoints(capsys):
    mock_remote, _ = _make_mock_remote(checkpoints=[])
    view_checkpoint_list(
        mock_remote, "proj123", "job456", SORT_BY_CREATED, SORT_ORDER_ASC
    )
    assert "No checkpoints found for job: job456." in capsys.readouterr().out


def test_view_checkpoint_list_sort_by_size(capsys):
    mock_remote, _ = _make_mock_remote(checkpoints=SAMPLE_CHECKPOINTS)
    view_checkpoint_list(mock_remote, "proj123", "job456", SORT_BY_SIZE, SORT_ORDER_ASC)
    out = capsys.readouterr().out
    assert out.find("ckpt-001") < out.find("ckpt-003") < out.find("ckpt-002")


def test_view_checkpoint_list_sort_by_size_desc(capsys):
    mock_remote, _ = _make_mock_remote(checkpoints=SAMPLE_CHECKPOINTS)
    view_checkpoint_list(
        mock_remote, "proj123", "job456", SORT_BY_SIZE, SORT_ORDER_DESC
    )
    out = capsys.readouterr().out
    assert out.find("ckpt-002") < out.find("ckpt-003") < out.find("ckpt-001")


def test_view_checkpoint_list_csv_format(capsys):
    mock_remote, _ = _make_mock_remote(checkpoints=SAMPLE_CHECKPOINTS)
    view_checkpoint_list(
        mock_remote,
        "proj123",
        "job456",
        SORT_BY_CREATED,
        SORT_ORDER_ASC,
        OUTPUT_FORMAT_CSV,
    )
    lines = capsys.readouterr().out.strip().split("\n")
    assert len(lines) == 4
    assert "Checkpoint ID" in lines[0]
    assert "ckpt-001" in lines[1]


def test_view_checkpoint_list_json_format(capsys):
    mock_remote, _ = _make_mock_remote(checkpoints=SAMPLE_CHECKPOINTS)
    view_checkpoint_list(
        mock_remote,
        "proj123",
        "job456",
        SORT_BY_CREATED,
        SORT_ORDER_ASC,
        OUTPUT_FORMAT_JSON,
    )
    output = json.loads(capsys.readouterr().out)
    assert output["job_id"] == "job456"
    assert output["total_checkpoints"] == 3
    ckpt_001 = next(
        c for c in output["checkpoints"] if c["checkpoint_id"] == "ckpt-001"
    )
    assert ckpt_001["lora_adapter_config"] == {"r": 16}
    ckpt_002 = next(
        c for c in output["checkpoints"] if c["checkpoint_id"] == "ckpt-002"
    )
    assert "lora_adapter_config" not in ckpt_002


def test_view_checkpoint_list_json_no_checkpoints(capsys):
    mock_remote, _ = _make_mock_remote(checkpoints=[])
    view_checkpoint_list(
        mock_remote,
        "proj123",
        "job456",
        SORT_BY_CREATED,
        SORT_ORDER_ASC,
        OUTPUT_FORMAT_JSON,
    )
    output = json.loads(capsys.readouterr().out)
    assert output["total_checkpoints"] == 0
    assert output["checkpoints"] == []


def test_view_checkpoint_list_api_error(capsys):
    mock_remote, mock_api = _make_mock_remote()
    mock_api.list_training_job_checkpoints.side_effect = Exception("API Error")
    with pytest.raises(Exception, match="API Error"):
        view_checkpoint_list(
            mock_remote, "proj123", "job456", SORT_BY_CREATED, SORT_ORDER_ASC
        )
    assert "Error fetching checkpoints: API Error" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# view_checkpoint_list — checkpoint_name scoping
# ---------------------------------------------------------------------------


def test_view_checkpoint_list_checkpoint_name_launches_explorer():
    mock_remote, _ = _make_mock_remote(
        checkpoints=SAMPLE_CHECKPOINTS, files=SAMPLE_FILES
    )
    with patch("truss.cli.train.checkpoint_viewer._explore_files") as mock_explore:
        view_checkpoint_list(
            mock_remote, "proj123", "job456", checkpoint_name="ckpt-001"
        )
    mock_explore.assert_called_once()
    assert mock_explore.call_args[1].get("initial_path") == "ckpt-001"


def test_view_checkpoint_list_checkpoint_name_passes_all_files_to_explorer():
    mock_remote, _ = _make_mock_remote(
        checkpoints=SAMPLE_CHECKPOINTS, files=SAMPLE_FILES
    )
    with patch("truss.cli.train.checkpoint_viewer._explore_files") as mock_explore:
        view_checkpoint_list(
            mock_remote, "proj123", "job456", checkpoint_name="ckpt-001"
        )
    assert len(mock_explore.call_args[0][0]) == len(SAMPLE_FILES)


def test_view_checkpoint_list_checkpoint_name_not_found(capsys):
    mock_remote, _ = _make_mock_remote(
        checkpoints=SAMPLE_CHECKPOINTS, files=SAMPLE_FILES
    )
    view_checkpoint_list(
        mock_remote, "proj123", "job456", checkpoint_name="nonexistent"
    )
    assert "No files found for checkpoint: nonexistent" in capsys.readouterr().out


def test_view_checkpoint_list_checkpoint_name_no_files(capsys):
    mock_remote, _ = _make_mock_remote(checkpoints=SAMPLE_CHECKPOINTS, files=[])
    view_checkpoint_list(mock_remote, "proj123", "job456", checkpoint_name="ckpt-001")
    assert "No files found" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# view_checkpoint_list — interactive flow
# ---------------------------------------------------------------------------


def test_view_checkpoint_list_interactive_no_checkpoints(capsys):
    mock_remote, _ = _make_mock_remote(checkpoints=[], files=[])
    view_checkpoint_list(mock_remote, "proj123", "job456", interactive=True)
    assert "No checkpoints found" in capsys.readouterr().out


def test_view_checkpoint_list_interactive_exit_from_selector():
    mock_remote, _ = _make_mock_remote(
        checkpoints=SAMPLE_CHECKPOINTS, files=SAMPLE_FILES
    )
    with patch(
        "truss.cli.train.checkpoint_viewer._select_checkpoint", return_value=None
    ):
        with patch("truss.cli.train.checkpoint_viewer._explore_files") as mock_explore:
            view_checkpoint_list(mock_remote, "proj123", "job456", interactive=True)
    mock_explore.assert_not_called()


def test_view_checkpoint_list_interactive_launches_explorer_for_selected():
    mock_remote, _ = _make_mock_remote(
        checkpoints=SAMPLE_CHECKPOINTS, files=SAMPLE_FILES
    )
    with patch(
        "truss.cli.train.checkpoint_viewer._select_checkpoint",
        side_effect=["ckpt-001", None],
    ):
        with patch("truss.cli.train.checkpoint_viewer._explore_files") as mock_explore:
            view_checkpoint_list(mock_remote, "proj123", "job456", interactive=True)
    mock_explore.assert_called_once()
    assert mock_explore.call_args[1].get("initial_path") == "ckpt-001"


def test_view_checkpoint_list_interactive_loops_back_to_selector():
    mock_remote, _ = _make_mock_remote(
        checkpoints=SAMPLE_CHECKPOINTS, files=SAMPLE_FILES
    )
    with patch(
        "truss.cli.train.checkpoint_viewer._select_checkpoint",
        side_effect=["ckpt-001", "ckpt-002", None],
    ):
        with patch("truss.cli.train.checkpoint_viewer._explore_files") as mock_explore:
            view_checkpoint_list(mock_remote, "proj123", "job456", interactive=True)
    assert mock_explore.call_count == 2
    paths = [c[1].get("initial_path") for c in mock_explore.call_args_list]
    assert paths == ["ckpt-001", "ckpt-002"]


def test_view_checkpoint_list_interactive_fetches_files_once():
    mock_remote, mock_api = _make_mock_remote(
        checkpoints=SAMPLE_CHECKPOINTS, files=SAMPLE_FILES
    )
    with patch(
        "truss.cli.train.checkpoint_viewer._select_checkpoint",
        side_effect=["ckpt-001", "ckpt-002", None],
    ):
        with patch("truss.cli.train.checkpoint_viewer._explore_files"):
            view_checkpoint_list(mock_remote, "proj123", "job456", interactive=True)
    assert mock_api.get_training_job_checkpoint_presigned_url.call_count == 1


# ---------------------------------------------------------------------------
# _fetch_and_display_file
# ---------------------------------------------------------------------------


@patch("truss.cli.train.checkpoint_viewer._open_in_pager")
@patch("truss.cli.train.checkpoint_viewer.requests.get")
def test_fetch_and_display_json_file(mock_get, mock_pager):
    mock_response = Mock()
    mock_response.text = '{"key": "value", "nested": {"a": 1}}'
    mock_response.raise_for_status = Mock()
    mock_get.return_value = mock_response
    _fetch_and_display_file(
        {
            "url": "https://example.com/config.json",
            "relative_file_name": "ckpt-001/adapter_config.json",
            "size_bytes": 100,
        }
    )
    content_arg = mock_pager.call_args[0][0]
    assert '"key": "value"' in content_arg


@patch("truss.cli.train.checkpoint_viewer._open_in_pager")
@patch("truss.cli.train.checkpoint_viewer.requests.get")
def test_fetch_and_display_text_file(mock_get, mock_pager):
    mock_response = Mock()
    mock_response.text = "some plain text content\nline 2"
    mock_response.raise_for_status = Mock()
    mock_get.return_value = mock_response
    _fetch_and_display_file(
        {
            "url": "https://example.com/readme.txt",
            "relative_file_name": "ckpt-001/readme.txt",
            "size_bytes": 100,
        }
    )
    assert "some plain text content" in mock_pager.call_args[0][0]


@patch("truss.cli.train.checkpoint_viewer.input")
@patch("truss.cli.train.checkpoint_viewer.requests.get")
def test_fetch_and_display_file_network_error_prompts_user(
    mock_get, mock_input, capsys
):
    mock_get.side_effect = requests.RequestException("connection refused")
    _fetch_and_display_file(
        {
            "url": "https://example.com/f",
            "relative_file_name": "ckpt-001/notes.txt",
            "size_bytes": 100,
        }
    )
    assert "Failed to fetch file" in capsys.readouterr().out
    mock_input.assert_called_once()


# ---------------------------------------------------------------------------
# _view_safetensor_file
# ---------------------------------------------------------------------------


@patch("truss.cli.train.checkpoint_viewer.input")
@patch("truss.cli.train.checkpoint_viewer._fetch_safetensor_header", return_value=None)
def test_view_safetensor_file_bad_header_prompts_user(_, mock_input, capsys):
    _view_safetensor_file(
        {
            "url": "https://example.com/m.safetensors",
            "relative_file_name": "ckpt-001/m.safetensors",
        }
    )
    assert "Failed to read safetensor header" in capsys.readouterr().out
    mock_input.assert_called_once()


# ---------------------------------------------------------------------------
# _select_checkpoint
# ---------------------------------------------------------------------------


def test_select_checkpoint_enter_returns_checkpoint_id():
    with patch("truss.cli.train.checkpoint_viewer._colored_fuzzy") as mock_fuzzy:
        mock_fuzzy.return_value.execute.return_value = ("checkpoint", "ckpt-001")
        result = _select_checkpoint(SAMPLE_CHECKPOINTS[:1], "job456")
    assert result == "ckpt-001"


def test_select_checkpoint_exit_returns_none():
    with patch("truss.cli.train.checkpoint_viewer._colored_fuzzy") as mock_fuzzy:
        mock_fuzzy.return_value.execute.return_value = ("exit", None)
        result = _select_checkpoint(SAMPLE_CHECKPOINTS[:1], "job456")
    assert result is None


def test_select_checkpoint_back_returns_none():
    with patch("truss.cli.train.checkpoint_viewer._colored_fuzzy") as mock_fuzzy:
        mock_fuzzy.return_value.execute.return_value = ("back", None)
        result = _select_checkpoint(SAMPLE_CHECKPOINTS[:1], "job456")
    assert result is None


def test_select_checkpoint_uses_allow_back_false():
    with patch("truss.cli.train.checkpoint_viewer._colored_fuzzy") as mock_fuzzy:
        mock_fuzzy.return_value.execute.return_value = ("exit", None)
        _select_checkpoint(SAMPLE_CHECKPOINTS[:1], "job456")
    assert mock_fuzzy.call_args[1].get("allow_back") is False


# ---------------------------------------------------------------------------
# _explore_files — initial_path normalization
# ---------------------------------------------------------------------------


def _make_explore_fuzzy_exit():
    """Return a _colored_fuzzy mock that immediately exits."""
    mock = Mock()
    mock.return_value.execute.return_value = ("exit", None)
    return mock


def test_explore_files_dot_initial_path_shows_root_files():
    """initial_path='.' should show root-level checkpoint directories, not be empty."""
    with patch(
        "truss.cli.train.checkpoint_viewer._colored_fuzzy", _make_explore_fuzzy_exit()
    ):
        # Should not raise and should reach the fuzzy prompt (not return immediately
        # due to an empty listing)
        _explore_files(SAMPLE_FILES, "job123", initial_path=".")


def test_explore_files_dot_initial_path_same_as_no_path():
    """initial_path='.' and initial_path=None should produce identical listings."""
    choices_dot = []
    choices_none = []

    def capture(target):
        def fuzzy(**kwargs):
            target.extend(kwargs.get("choices", []))
            m = Mock()
            m.execute.return_value = ("exit", None)
            return m

        return fuzzy

    with patch(
        "truss.cli.train.checkpoint_viewer._colored_fuzzy", capture(choices_dot)
    ):
        _explore_files(SAMPLE_FILES, "job123", initial_path=".")

    with patch(
        "truss.cli.train.checkpoint_viewer._colored_fuzzy", capture(choices_none)
    ):
        _explore_files(SAMPLE_FILES, "job123", initial_path=None)

    assert choices_dot == choices_none


def test_explore_files_dot_slash_prefix_stripped():
    """initial_path='./ckpt-001' should start inside ckpt-001, not be empty."""
    choices_captured = []

    def fuzzy(**kwargs):
        choices_captured.extend(kwargs.get("choices", []))
        m = Mock()
        m.execute.return_value = ("exit", None)
        return m

    with patch("truss.cli.train.checkpoint_viewer._colored_fuzzy", fuzzy):
        _explore_files(SAMPLE_FILES, "job123", initial_path="./ckpt-001")

    # Should show files inside ckpt-001 (adapter_model.safetensors, adapter_config.json)
    names = [
        c["name"] for c in choices_captured if c.get("value", (None,))[0] != "exit"
    ]
    assert any("adapter" in n for n in names)


def test_explore_files_normal_initial_path_unaffected():
    """A normal initial_path like 'ckpt-001' should still work correctly."""
    choices_captured = []

    def fuzzy(**kwargs):
        choices_captured.extend(kwargs.get("choices", []))
        m = Mock()
        m.execute.return_value = ("exit", None)
        return m

    with patch("truss.cli.train.checkpoint_viewer._colored_fuzzy", fuzzy):
        _explore_files(SAMPLE_FILES, "job123", initial_path="ckpt-001")

    names = [
        c["name"] for c in choices_captured if c.get("value", (None,))[0] != "exit"
    ]
    assert any("adapter" in n for n in names)
