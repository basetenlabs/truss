from unittest.mock import Mock, patch

import pytest

from truss.cli.train.core import ACTIVE_JOB_STATUSES, QUEUED_JOB_STATUSES
from truss.cli.train.poller import JOB_STARTING_STATES, TrainingPollerMixin


def _make_poller(statuses: list[str]) -> TrainingPollerMixin:
    """Create a TrainingPollerMixin with a mocked API returning the given statuses in sequence."""
    mock_api = Mock()
    mock_api.get_training_job.side_effect = [
        {"training_job": {"current_status": s, "error_message": None}} for s in statuses
    ]
    poller = TrainingPollerMixin.__new__(TrainingPollerMixin)
    TrainingPollerMixin.__init__(poller, mock_api, "proj1", "job1")
    return poller


def test_pending_in_job_starting_states():
    assert "TRAINING_JOB_PENDING" in JOB_STARTING_STATES


def test_starting_status_message_pending():
    poller = _make_poller(["TRAINING_JOB_PENDING"])
    poller._update_from_current_status()
    assert "GPU capacity" in poller._starting_status_message()
    assert "queued" in poller._starting_status_message()


def test_starting_status_message_created():
    poller = _make_poller(["TRAINING_JOB_CREATED"])
    poller._update_from_current_status()
    assert "deploy" in poller._starting_status_message()
    assert "TRAINING_JOB_CREATED" in poller._starting_status_message()


@patch("truss.cli.train.poller.time.sleep")
def test_before_polling_waits_through_pending(mock_sleep):
    """before_polling should spin through PENDING and CREATED before exiting."""
    statuses = [
        "TRAINING_JOB_PENDING",
        "TRAINING_JOB_PENDING",
        "TRAINING_JOB_CREATED",
        "TRAINING_JOB_DEPLOYING",
    ]
    poller = _make_poller(statuses)
    poller.before_polling()
    # Should have exited once status left JOB_STARTING_STATES
    assert poller._current_status.status == "TRAINING_JOB_DEPLOYING"


def test_pending_not_in_active_job_statuses():
    assert "TRAINING_JOB_PENDING" not in ACTIVE_JOB_STATUSES


def test_pending_in_queued_job_statuses():
    assert "TRAINING_JOB_PENDING" in QUEUED_JOB_STATUSES


def test_view_training_details_splits_queued_and_active(capsys):
    """Queued (PENDING) jobs appear under 'Queued' header, not 'Active'."""
    from truss.cli.train.core import view_training_details

    mock_remote = Mock()
    mock_remote.remote_url = "https://app.baseten.co"
    mock_remote.api.list_training_projects.return_value = []
    ts = "2024-01-01T00:00:00+00:00"
    mock_remote.api.search_training_jobs.return_value = [
        {
            "id": "q1",
            "current_status": "TRAINING_JOB_PENDING",
            "name": "queued-job",
            "training_project": {"id": "p1", "name": "proj"},
            "instance_type": {"name": "H100"},
            "created_at": ts,
            "updated_at": ts,
        },
        {
            "id": "a1",
            "current_status": "TRAINING_JOB_RUNNING",
            "name": "active-job",
            "training_project": {"id": "p1", "name": "proj"},
            "instance_type": {"name": "H100"},
            "created_at": ts,
            "updated_at": ts,
        },
    ]

    view_training_details(mock_remote, project_id=None, job_id=None)
    captured = capsys.readouterr()
    # Queued section renders as a table with its own title and columns
    assert "Queued Training Jobs" in captured.out
    assert "q1" in captured.out
    # Active section renders as a table with a Status column
    assert "Active Training Jobs" in captured.out
    assert "a1" in captured.out


def test_view_training_details_no_jobs_shows_fallback(capsys):
    """When both queued and active are empty, show the 'no active' message."""
    from truss.cli.train.core import view_training_details

    mock_remote = Mock()
    mock_remote.remote_url = "https://app.baseten.co"
    mock_remote.api.list_training_projects.return_value = []
    mock_remote.api.search_training_jobs.return_value = []

    view_training_details(mock_remote, project_id=None, job_id=None)
    captured = capsys.readouterr()
    assert "No queued or active training jobs" in captured.out


@pytest.mark.parametrize(
    "status,expected_fragment",
    [("TRAINING_JOB_PENDING", "pending"), ("TRAINING_JOB_QUEUED", "queued")],
)
def test_handle_post_create_logic_queue_messages(status, expected_fragment, capsys):
    """_handle_post_create_logic should print queue/pending message for non-deployed statuses."""
    from truss.cli.train_commands import _handle_post_create_logic

    mock_remote = Mock()
    mock_remote.remote_url = "https://app.baseten.co"
    job_resp = {
        "id": "job123",
        "current_status": status,
        "training_project": {"id": "proj456", "name": "my-project"},
    }
    # tail=False so we don't try to stream logs
    _handle_post_create_logic(job_resp, mock_remote, tail=False)
    captured = capsys.readouterr()
    assert expected_fragment in captured.out.lower()
