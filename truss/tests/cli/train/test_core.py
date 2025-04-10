from io import StringIO
from unittest.mock import Mock, patch

import pytest
from rich.console import Console

from truss.cli.train.core import view_training_job_metrics

def test_view_training_job_metrics():
    # Create a console that writes to StringIO for capture
    string_io = StringIO()
    console = Console(file=string_io)
    
    # Mock the remote provider and its API
    mock_api = Mock()
    mock_remote = Mock()
    mock_remote.api = mock_api
    
    # Mock the MetricsWatcher
    with patch('truss.cli.train.core.MetricsWatcher') as mock_metrics_watcher:
        # Set up mock API responses for get_args_for_logs
        mock_api.search_training_jobs.return_value = [{
            'id': 'job123',
            'training_project': {'id': 'proj456'}
        }]
        
        # Call the function
        view_training_job_metrics(
            console=console,
            remote_provider=mock_remote,
            project_id=None,
            job_id=None
        )
        
        # Verify MetricsWatcher was created with correct args
        mock_metrics_watcher.assert_called_once_with(
            mock_api,
            'proj456',
            'job123',
            console
        )
        
        # Verify display_live_metrics was called
        mock_metrics_watcher.return_value.display_live_metrics.assert_called_once()