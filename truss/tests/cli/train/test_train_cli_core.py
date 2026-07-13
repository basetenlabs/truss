from unittest.mock import Mock, patch

import click
import pytest
from rich.console import Console

from truss.cli.train.cache import (
    calculate_directory_sizes,
    create_file_summary_with_directory_sizes,
)
from truss.cli.train.core import (
    _format_capacity_type,
    display_queued_jobs,
    display_training_capacity,
    display_training_jobs,
    update_team_training_gpu_capacity,
    view_training_job_metrics,
)
from truss.remote.baseten.custom_types import FileSummary, TeamType


@patch("truss.cli.train.metrics_watcher.time.sleep")
@patch(
    "truss.cli.train.poller.JOB_TERMINATION_GRACE_PERIOD_SEC", -1
)  # don't perform cleanup
def test_view_training_job_metrics(time_sleep, capfd):
    # Mock the remote provider and its API
    mock_api = Mock()
    mock_remote = Mock()
    mock_remote.api = mock_api

    # Set up mock API responses for get_args_for_monitoring
    mock_api.search_training_jobs.return_value = [
        {"id": "job123", "training_project": {"id": "proj456"}}
    ]
    mock_api.get_training_job.side_effect = [
        {
            "training_job": {
                "id": "job123",
                "training_project": {"id": "proj456"},
                "current_status": "TRAINING_JOB_RUNNING",
            }
        },
        {
            "training_job": {
                "id": "job123",
                "training_project": {"id": "proj456"},
                "current_status": "TRAINING_JOB_COMPLETED",
            }
        },
    ]

    mock_api.get_training_job_metrics.side_effect = [
        {
            "training_job": {
                "id": "job123",
                "training_project": {"id": "proj456"},
                "current_status": "TRAINING_JOB_RUNNING",
            },
            "cpu_usage": [{"timestamp": "", "value": 3.2}],
            "cpu_memory_usage_bytes": [{"timestamp": "", "value": 1234}],
            "gpu_utilization": {
                "0": [{"timestamp": "", "value": 0.2}],
                "1": [{"timestamp": "", "value": 0.3}],
            },
            "gpu_memory_usage_bytes": {
                "0": [{"timestamp": "", "value": 4321}],
                "1": [{"timestamp": "", "value": 2222}],
            },
            "per_node_metrics": [
                {
                    "node_id": "node-0",
                    "metrics": {
                        "cpu_usage": [{"timestamp": "", "value": 3.2}],
                        "cpu_memory_usage_bytes": [{"timestamp": "", "value": 1234}],
                        "gpu_utilization": {
                            "0": [{"timestamp": "", "value": 0.2}],
                            "1": [{"timestamp": "", "value": 0.3}],
                        },
                        "gpu_memory_usage_bytes": {
                            "0": [{"timestamp": "", "value": 4321}],
                            "1": [{"timestamp": "", "value": 2222}],
                        },
                    },
                },
                {
                    "node_id": "node-1",
                    "metrics": {
                        "cpu_usage": [{"timestamp": "", "value": 2.8}],
                        "cpu_memory_usage_bytes": [{"timestamp": "", "value": 1000}],
                        "gpu_utilization": {
                            "0": [{"timestamp": "", "value": 0.15}],
                            "1": [{"timestamp": "", "value": 0.25}],
                        },
                        "gpu_memory_usage_bytes": {
                            "0": [{"timestamp": "", "value": 4000}],
                            "1": [{"timestamp": "", "value": 2000}],
                        },
                    },
                },
            ],
        },
        {
            "training_job": {
                "id": "job123",
                "training_project": {"id": "proj456"},
                "current_status": "TRAINING_JOB_COMPLETED",
            },
            "cpu_usage": [{"timestamp": "", "value": 3.2}],
            "cpu_memory_usage_bytes": [{"timestamp": "", "value": 1234}],
            "gpu_utilization": {
                "0": [{"timestamp": "", "value": 0.2}],
                "1": [{"timestamp": "", "value": 0.3}],
            },
            "gpu_memory_usage_bytes": {
                "0": [{"timestamp": "", "value": 4321}],
                "1": [{"timestamp": "", "value": 2222}],
            },
            "per_node_metrics": [
                {
                    "node_id": "node-0",
                    "metrics": {
                        "cpu_usage": [{"timestamp": "", "value": 3.2}],
                        "cpu_memory_usage_bytes": [{"timestamp": "", "value": 1234}],
                        "gpu_utilization": {
                            "0": [{"timestamp": "", "value": 0.2}],
                            "1": [{"timestamp": "", "value": 0.3}],
                        },
                        "gpu_memory_usage_bytes": {
                            "0": [{"timestamp": "", "value": 4321}],
                            "1": [{"timestamp": "", "value": 2222}],
                        },
                    },
                },
                {
                    "node_id": "node-1",
                    "metrics": {
                        "cpu_usage": [{"timestamp": "", "value": 2.8}],
                        "cpu_memory_usage_bytes": [{"timestamp": "", "value": 1000}],
                        "gpu_utilization": {
                            "0": [{"timestamp": "", "value": 0.15}],
                            "1": [{"timestamp": "", "value": 0.25}],
                        },
                        "gpu_memory_usage_bytes": {
                            "0": [{"timestamp": "", "value": 4000}],
                            "1": [{"timestamp": "", "value": 2000}],
                        },
                    },
                },
            ],
        },
        {
            "training_job": {
                "id": "job123",
                "training_project": {"id": "proj456"},
                "current_status": "TRAINING_JOB_COMPLETED",
            },
            "cpu_usage": [{"timestamp": "", "value": 3.2}],
            "cpu_memory_usage_bytes": [{"timestamp": "", "value": 1234}],
            "gpu_utilization": {
                "0": [{"timestamp": "", "value": 0.2}],
                "1": [{"timestamp": "", "value": 0.3}],
            },
            "gpu_memory_usage_bytes": {
                "0": [{"timestamp": "", "value": 4321}],
                "1": [{"timestamp": "", "value": 2222}],
            },
            "per_node_metrics": [
                {
                    "node_id": "node-0",
                    "metrics": {
                        "cpu_usage": [{"timestamp": "", "value": 3.2}],
                        "cpu_memory_usage_bytes": [{"timestamp": "", "value": 1234}],
                        "gpu_utilization": {
                            "0": [{"timestamp": "", "value": 0.2}],
                            "1": [{"timestamp": "", "value": 0.3}],
                        },
                        "gpu_memory_usage_bytes": {
                            "0": [{"timestamp": "", "value": 4321}],
                            "1": [{"timestamp": "", "value": 2222}],
                        },
                    },
                },
                {
                    "node_id": "node-1",
                    "metrics": {
                        "cpu_usage": [{"timestamp": "", "value": 2.8}],
                        "cpu_memory_usage_bytes": [{"timestamp": "", "value": 1000}],
                        "gpu_utilization": {
                            "0": [{"timestamp": "", "value": 0.15}],
                            "1": [{"timestamp": "", "value": 0.25}],
                        },
                        "gpu_memory_usage_bytes": {
                            "0": [{"timestamp": "", "value": 4000}],
                            "1": [{"timestamp": "", "value": 2002}],
                        },
                    },
                },
            ],
        },
    ]

    # Call the function
    view_training_job_metrics(remote_provider=mock_remote, project_id=None, job_id=None)
    out, err = capfd.readouterr()
    assert "Training job completed successfully" in out
    assert "Error fetching metrics" not in out


def test_calculate_directory_sizes():
    """Test calculate_directory_sizes function with various file structures."""
    # Create test files with a nested directory structure
    files = [
        FileSummary(
            path="/root",
            size_bytes=0,
            modified="2023-01-01T00:00:00Z",
            file_type="directory",
            permissions="drwxr-xr-x",
        ),
        FileSummary(
            path="/root/file1.txt",
            size_bytes=100,
            modified="2023-01-01T00:00:00Z",
            file_type="file",
            permissions="-rw-r--r--",
        ),
        FileSummary(
            path="/root/subdir",
            size_bytes=0,
            modified="2023-01-01T00:00:00Z",
            file_type="directory",
            permissions="drwxr-xr-x",
        ),
        FileSummary(
            path="/root/subdir/file2.txt",
            size_bytes=200,
            modified="2023-01-01T00:00:00Z",
            file_type="file",
            permissions="-rw-r--r--",
        ),
        FileSummary(
            path="/root/subdir/file3.txt",
            size_bytes=300,
            modified="2023-01-01T00:00:00Z",
            file_type="file",
            permissions="-rw-r--r--",
        ),
        FileSummary(
            path="/root/other_file.txt",
            size_bytes=50,
            modified="2023-01-01T00:00:00Z",
            file_type="file",
            permissions="-rw-r--r--",
        ),
    ]

    result = calculate_directory_sizes(files)

    # Check that directory sizes are calculated correctly
    assert result["/root/subdir"] == 500  # 200 + 300
    assert result["/root"] == 650  # 100 + 200 + 300 + 50

    # Check that files are not included in the result (only directories)
    assert "/root/file1.txt" not in result
    assert "/root/subdir/file2.txt" not in result
    assert "/root/subdir/file3.txt" not in result
    assert "/root/other_file.txt" not in result


def test_calculate_directory_sizes_empty_list():
    """Test calculate_directory_sizes with empty file list."""
    result = calculate_directory_sizes([])
    assert result == {}


def test_calculate_directory_sizes_no_directories():
    """Test calculate_directory_sizes with only files (no directories)."""
    files = [
        FileSummary(
            path="/file1.txt",
            size_bytes=100,
            modified="2023-01-01T00:00:00Z",
            file_type="file",
            permissions="-rw-r--r--",
        ),
        FileSummary(
            path="/file2.txt",
            size_bytes=200,
            modified="2023-01-01T00:00:00Z",
            file_type="file",
            permissions="-rw-r--r--",
        ),
    ]

    result = calculate_directory_sizes(files)
    assert result == {}


def test_create_file_summary_with_directory_sizes():
    """Test create_file_summary_with_directory_sizes function."""
    files = [
        FileSummary(
            path="/root",
            size_bytes=0,
            modified="2023-01-01T00:00:00Z",
            file_type="directory",
            permissions="drwxr-xr-x",
        ),
        FileSummary(
            path="/root/file1.txt",
            size_bytes=100,
            modified="2023-01-01T00:00:00Z",
            file_type="file",
            permissions="-rw-r--r--",
        ),
        FileSummary(
            path="/root/subdir",
            size_bytes=0,
            modified="2023-01-01T00:00:00Z",
            file_type="directory",
            permissions="drwxr-xr-x",
        ),
        FileSummary(
            path="/root/subdir/file2.txt",
            size_bytes=200,
            modified="2023-01-01T00:00:00Z",
            file_type="file",
            permissions="-rw-r--r--",
        ),
    ]

    result = create_file_summary_with_directory_sizes(files)

    # Check that we get the correct number of FileSummaryWithTotalSize objects
    assert len(result) == 4

    # Check that files have their original size as total_size
    file1_summary = next(f for f in result if f.file_summary.path == "/root/file1.txt")
    assert file1_summary.total_size == 100

    file2_summary = next(
        f for f in result if f.file_summary.path == "/root/subdir/file2.txt"
    )
    assert file2_summary.total_size == 200

    # Check that directories have calculated total sizes
    subdir_summary = next(f for f in result if f.file_summary.path == "/root/subdir")
    assert subdir_summary.total_size == 200  # Only file2.txt

    root_summary = next(f for f in result if f.file_summary.path == "/root")
    assert root_summary.total_size == 300  # file1.txt + file2.txt


def test_create_file_summary_with_directory_sizes_empty_list():
    """Test create_file_summary_with_directory_sizes with empty file list."""
    result = create_file_summary_with_directory_sizes([])
    assert result == []


def test_calculate_directory_sizes_max_depth():
    """Test that calculate_directory_sizes respects the max_depth parameter.

    The max_depth parameter controls how many parent directories up from each file
    the algorithm will traverse to add the file's size to parent directories.
    """
    # Create a deep directory structure: /root/level1/level2/level3/level4/level5/file.txt
    files = [
        # Root directory
        FileSummary(
            path="/root",
            size_bytes=0,
            modified="2023-01-01T00:00:00Z",
            file_type="directory",
            permissions="drwxr-xr-x",
        ),
        # Level 1 directory
        FileSummary(
            path="/root/level1",
            size_bytes=0,
            modified="2023-01-01T00:00:00Z",
            file_type="directory",
            permissions="drwxr-xr-x",
        ),
        # Level 2 directory
        FileSummary(
            path="/root/level1/level2",
            size_bytes=0,
            modified="2023-01-01T00:00:00Z",
            file_type="directory",
            permissions="drwxr-xr-x",
        ),
        # Level 3 directory
        FileSummary(
            path="/root/level1/level2/level3",
            size_bytes=0,
            modified="2023-01-01T00:00:00Z",
            file_type="directory",
            permissions="drwxr-xr-x",
        ),
        # Level 4 directory
        FileSummary(
            path="/root/level1/level2/level3/level4",
            size_bytes=0,
            modified="2023-01-01T00:00:00Z",
            file_type="directory",
            permissions="drwxr-xr-x",
        ),
        # Level 5 directory
        FileSummary(
            path="/root/level1/level2/level3/level4/level5",
            size_bytes=0,
            modified="2023-01-01T00:00:00Z",
            file_type="directory",
            permissions="drwxr-xr-x",
        ),
        # File at level 1
        FileSummary(
            path="/root/level1/file1.txt",
            size_bytes=100,
            modified="2023-01-01T00:00:00Z",
            file_type="file",
            permissions="-rw-r--r--",
        ),
        # File at level 2
        FileSummary(
            path="/root/level1/level2/file2.txt",
            size_bytes=200,
            modified="2023-01-01T00:00:00Z",
            file_type="file",
            permissions="-rw-r--r--",
        ),
        # File at level 3
        FileSummary(
            path="/root/level1/level2/level3/file3.txt",
            size_bytes=300,
            modified="2023-01-01T00:00:00Z",
            file_type="file",
            permissions="-rw-r--r--",
        ),
    ]

    result_depth_0 = calculate_directory_sizes(files, max_depth=0)
    assert result_depth_0["/root"] == 0
    assert result_depth_0["/root/level1"] == 0
    assert result_depth_0["/root/level1/level2"] == 0
    assert result_depth_0["/root/level1/level2/level3"] == 0

    # ensure that we stop early if the max depth is reached
    result_depth_2 = calculate_directory_sizes(files, max_depth=2)

    assert result_depth_2["/root"] == 0
    assert result_depth_2["/root/level1"] == 100  # file1.txt only
    assert result_depth_2["/root/level1/level2"] == 200  # file2.txt only
    assert result_depth_2["/root/level1/level2/level3"] == 300  # file3.txt only


def _capacity_rows(out: str) -> dict:
    return {
        cells[0]: cells
        for line in out.splitlines()
        if (cells := [c.strip() for c in line.split("│") if c.strip()])
    }


def test_display_training_capacity(capsys):
    """Usage is split into On-Demand and Spot columns per GPU type."""
    mock_api = Mock()
    mock_api.get_training_capacity.return_value = {
        "gpu_capacities": [
            {
                "gpu_type": "H100",
                "baseline": 16,
                "limit": 64,
                "usage_count": 40,
                "dedicated_usage_count": 24,
                "spot_usage_count": 16,
            },
            {
                "gpu_type": "A10G",
                "baseline": 0,
                "limit": 8,
                "usage_count": 0,
                "dedicated_usage_count": 0,
                "spot_usage_count": 0,
            },
        ]
    }
    mock_remote = Mock()
    mock_remote.api = mock_api

    with patch("truss.cli.train.core.console", Console(width=200)):
        display_training_capacity(mock_remote)

    rows = _capacity_rows(capsys.readouterr().out)
    # GPU Type, Baseline, Limit, On-Demand, Spot
    assert rows["H100"] == ["H100", "16", "64", "24", "16"]
    assert rows["A10G"] == ["A10G", "0", "8", "0", "0"]


def test_display_training_capacity_with_teams(capsys):
    """Team capacities are rendered in a second table with split usage columns."""
    mock_api = Mock()
    mock_api.get_training_capacity.return_value = {
        "gpu_capacities": [
            {
                "gpu_type": "H100",
                "baseline": 16,
                "limit": 64,
                "usage_count": 32,
                "dedicated_usage_count": 32,
                "spot_usage_count": 0,
            }
        ],
        "team_gpu_capacities": [
            {
                "team_id": "team_abc",
                "team_name": "ml-research",
                "gpu_type": "H100",
                "baseline": 0,
                "limit": 32,
                "usage_count": 8,
                "dedicated_usage_count": 2,
                "spot_usage_count": 6,
            }
        ],
    }
    mock_remote = Mock()
    mock_remote.api = mock_api

    with patch("truss.cli.train.core.console", Console(width=200)):
        display_training_capacity(mock_remote)

    out = capsys.readouterr().out
    rows = _capacity_rows(out)

    assert "Team Training GPU Capacity" in out
    # Org table keeps Baseline: GPU Type, Baseline, Limit, On-Demand, Spot
    assert rows["H100"] == ["H100", "16", "64", "32", "0"]
    # Team table dropped Baseline (#2539): Team, GPU Type, Limit, On-Demand, Spot
    assert rows["ml-research"] == ["ml-research", "H100", "32", "2", "6"]
    assert "Baseline" not in rows["Team"]


def test_display_training_capacity_pre_split_backend(capsys):
    """A pre-split backend (only usage_count) shows the total under On-Demand."""
    mock_api = Mock()
    mock_api.get_training_capacity.return_value = {
        "gpu_capacities": [
            {"gpu_type": "H100", "baseline": 16, "limit": 64, "usage_count": 32}
        ]
    }
    mock_remote = Mock()
    mock_remote.api = mock_api

    with patch("truss.cli.train.core.console", Console(width=200)):
        display_training_capacity(mock_remote)

    rows = _capacity_rows(capsys.readouterr().out)
    assert rows["H100"] == ["H100", "16", "64", "32", "0"]


def test_display_training_capacity_teams_only(capsys):
    """A team table is shown even when there are no org-level capacities."""
    mock_api = Mock()
    mock_api.get_training_capacity.return_value = {
        "gpu_capacities": [],
        "team_gpu_capacities": [
            {
                "team_id": "team_abc",
                "team_name": "ml-research",
                "gpu_type": "H100",
                "baseline": 0,
                "limit": 32,
                "usage_count": 0,
            }
        ],
    }
    mock_remote = Mock()
    mock_remote.api = mock_api

    display_training_capacity(mock_remote)

    out = capsys.readouterr().out
    assert "Team Training GPU Capacity" in out
    assert "No Training GPU capacity limits." not in out


def test_display_training_capacity_empty(capsys):
    """Empty state message is printed when there are no GPU capacities."""
    mock_api = Mock()
    mock_api.get_training_capacity.return_value = {
        "gpu_capacities": [],
        "team_gpu_capacities": [],
    }
    mock_remote = Mock()
    mock_remote.api = mock_api

    display_training_capacity(mock_remote)

    assert "No Training GPU capacity limits." in capsys.readouterr().out


def test_update_team_training_gpu_capacity_resolves_team_and_calls_api():
    """Team name is resolved to a team_id before calling the update API."""
    mock_api = Mock()
    mock_api.get_teams.return_value = {
        "ml-research": TeamType(id="team_abc", name="ml-research", default=False)
    }
    mock_api.update_team_training_gpu_capacity.return_value = {
        "team_id": "team_abc",
        "team_name": "ml-research",
        "gpu_type": "H100",
        "baseline": 0,
        "limit": 32,
        "usage_count": 0,
    }
    mock_remote = Mock()
    mock_remote.api = mock_api

    result = update_team_training_gpu_capacity(
        mock_remote, team_name="ml-research", gpu_type="H100", capacity=32
    )

    mock_api.update_team_training_gpu_capacity.assert_called_once_with(
        team_id="team_abc", gpu_type="H100", max_gpus=32
    )
    assert result["limit"] == 32


def test_update_team_training_gpu_capacity_unknown_team_raises():
    """An unknown team name raises a ClickException listing the available teams."""
    mock_api = Mock()
    mock_api.get_teams.return_value = {
        "ml-research": TeamType(id="team_abc", name="ml-research", default=False)
    }
    mock_remote = Mock()
    mock_remote.api = mock_api

    with pytest.raises(click.ClickException, match="ml-research"):
        update_team_training_gpu_capacity(
            mock_remote, team_name="does-not-exist", gpu_type="H100", capacity=32
        )

    mock_api.update_team_training_gpu_capacity.assert_not_called()


def test_format_capacity_type():
    """availability_model maps to human-readable labels; absent reads as on-demand."""
    assert _format_capacity_type({"availability_model": "spot"}) == "Spot"
    assert _format_capacity_type({"availability_model": "dedicated"}) == "On-demand"
    # Absent / null predates the field and reads as on-demand.
    assert _format_capacity_type({}) == "On-demand"
    assert _format_capacity_type({"availability_model": None}) == "On-demand"
    # An unknown future value is surfaced rather than dropped.
    assert _format_capacity_type({"availability_model": "reserved"}) == "Reserved"


def _make_job(job_id, availability_model=None, **overrides):
    job = {
        "id": job_id,
        "name": f"{job_id}-name",
        "training_project": {"id": "proj1", "name": "proj"},
        "instance_type": {"name": "H100"},
        "current_status": "TRAINING_JOB_RUNNING",
        "priority": 0,
        "created_at": "2026-07-13T00:00:00Z",
        "user": {"email": "a@b.co"},
    }
    if availability_model is not None:
        job["availability_model"] = availability_model
    job.update(overrides)
    return job


def test_display_queued_jobs_shows_capacity_type(capsys):
    """Queued jobs table includes a Capacity Type column mapped from availability_model."""
    jobs = [
        _make_job("spotjob", availability_model="spot"),
        _make_job("dedjob", availability_model="dedicated"),
        _make_job("oldjob"),  # no availability_model -> on-demand
    ]

    # Render wide so the extra column isn't truncated at the default 80 cols.
    with patch("truss.cli.train.core.console", Console(width=200)):
        display_queued_jobs(jobs, "https://app.baseten.co")

    out = capsys.readouterr().out
    assert "Capacity Type" in out
    assert "Spot" in out
    assert "On-demand" in out


def test_display_training_jobs_shows_capacity_type(capsys):
    """Active jobs table includes a Capacity Type column mapped from availability_model."""
    jobs = [
        _make_job("spotjob", availability_model="spot"),
        _make_job("dedjob", availability_model="dedicated"),
    ]

    with patch("truss.cli.train.core.console", Console(width=200)):
        display_training_jobs(jobs, "https://app.baseten.co")

    out = capsys.readouterr().out
    assert "Capacity Type" in out
    assert "Spot" in out
    assert "On-demand" in out
