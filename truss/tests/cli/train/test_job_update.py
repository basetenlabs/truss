from unittest.mock import MagicMock

import pytest

from truss.cli.train import core as train_cli
from truss_train.definitions import AvailabilityModel


@pytest.fixture
def mock_remote():
    remote = MagicMock()
    remote.api = MagicMock()
    return remote


class TestUpdateTrainingJob:
    """Test cases for the update_training_job function."""

    def test_update_training_job_success(self, mock_remote):
        """Test updating a training job with a specific job ID."""
        mock_remote.api.search_training_jobs.return_value = [
            {
                "id": "test_job_123",
                "training_project": {"id": "project_456", "name": "test-project"},
            }
        ]
        mock_remote.api.update_training_job.return_value = {
            "id": "test_job_123",
            "priority": 42,
            "training_project": {"id": "project_456", "name": "test-project"},
        }

        result = train_cli.update_training_job(
            remote_provider=mock_remote, job_id="test_job_123", priority=42
        )

        assert result["id"] == "test_job_123"
        assert result["priority"] == 42

        mock_remote.api.search_training_jobs.assert_called_once_with(
            job_id="test_job_123"
        )
        mock_remote.api.update_training_job.assert_called_once_with(
            "project_456", "test_job_123", priority=42, availability_model=None
        )

    def test_update_training_job_availability_model(self, mock_remote):
        """The availability model is passed to the API as its wire value."""
        mock_remote.api.search_training_jobs.return_value = [
            {
                "id": "test_job_123",
                "training_project": {"id": "project_456", "name": "test-project"},
            }
        ]
        mock_remote.api.update_training_job.return_value = {"id": "test_job_123"}

        train_cli.update_training_job(
            remote_provider=mock_remote,
            job_id="test_job_123",
            availability_model=AvailabilityModel.SPOT,
        )

        mock_remote.api.update_training_job.assert_called_once_with(
            "project_456", "test_job_123", priority=None, availability_model="spot"
        )

    def test_update_training_job_priority_and_availability_model(self, mock_remote):
        """Both fields are forwarded together when both are provided."""
        mock_remote.api.search_training_jobs.return_value = [
            {
                "id": "test_job_123",
                "training_project": {"id": "project_456", "name": "test-project"},
            }
        ]
        mock_remote.api.update_training_job.return_value = {"id": "test_job_123"}

        train_cli.update_training_job(
            remote_provider=mock_remote,
            job_id="test_job_123",
            priority=7,
            availability_model=AvailabilityModel.DEDICATED,
        )

        mock_remote.api.update_training_job.assert_called_once_with(
            "project_456", "test_job_123", priority=7, availability_model="dedicated"
        )

    def test_update_training_job_no_job_found(self, mock_remote):
        """Test updating a non-existent job ID."""
        mock_remote.api.search_training_jobs.return_value = []

        with pytest.raises(
            RuntimeError, match="No training job found with ID: nonexistent_job"
        ):
            train_cli.update_training_job(
                remote_provider=mock_remote, job_id="nonexistent_job", priority=42
            )

        mock_remote.api.update_training_job.assert_not_called()

    def test_update_training_job_no_fields_raises(self, mock_remote):
        """Test that updating with no fields provided raises an error."""
        with pytest.raises(
            ValueError, match="At least one field to update must be provided"
        ):
            train_cli.update_training_job(
                remote_provider=mock_remote, job_id="test_job_123"
            )

        mock_remote.api.search_training_jobs.assert_not_called()
        mock_remote.api.update_training_job.assert_not_called()
