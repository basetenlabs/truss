from io import StringIO
import os
from pathlib import Path
from truss.cli.train.core import DeployCheckpointTemplatingArgs, render_vllm_lora_truss_config
from truss.base import truss_config
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

def test_render_vllm_lora_truss_config():
    template_args = DeployCheckpointTemplatingArgs(
        checkpoint_id="checkpoint-1",
        base_model_id="google/gemma-3-27b-it",
        hf_secret_name="hf_access_token",
        dtype="bfloat16",
        model_name="gemma-3-27b-it-vLLM-LORA",
        accelerator=truss_config.AcceleratorSpec(accelerator="H100", count=4),
    )
    rendered_truss = render_vllm_lora_truss_config(template_args)
    test_truss = truss_config.TrussConfig.from_yaml(Path(os.path.dirname(__file__), "resources/test_deploy_from_checkpoint_config.yml"))
    assert test_truss.model_name == rendered_truss.model_name
    assert test_truss.training_checkpoints.checkpoints[0].id == rendered_truss.training_checkpoints.checkpoints[0].id
    assert test_truss.training_checkpoints.checkpoints[0].name == rendered_truss.training_checkpoints.checkpoints[0].name
    assert test_truss.docker_server.start_command == rendered_truss.docker_server.start_command
    assert test_truss.resources.accelerator == rendered_truss.resources.accelerator
    assert test_truss.secrets == rendered_truss.secrets
    assert test_truss.training_checkpoints == rendered_truss.training_checkpoints
