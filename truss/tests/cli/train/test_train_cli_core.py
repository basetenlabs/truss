from io import StringIO
from unittest.mock import Mock, patch

from rich.console import Console

from truss.cli.train.core import view_training_job_metrics


@patch("truss.cli.train.metrics_watcher.time.sleep")
@patch(
    "truss.cli.train.poller.JOB_TERMINATION_GRACE_PERIOD_SEC", -1
)  # don't perform cleanup
def test_view_training_job_metrics(time_sleep):
    # Create a console that writes to StringIO for capture
    string_io = StringIO()
    console = Console(file=string_io)

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
        },
    ]

    # Call the function
    view_training_job_metrics(
        console=console, remote_provider=mock_remote, project_id=None, job_id=None
    )
    assert "Training job completed successfully" in string_io.getvalue()
    assert "Error fetching metrics" not in string_io.getvalue()
