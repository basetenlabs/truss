from unittest.mock import MagicMock, patch

import click
import pytest

from truss.cli.train import core as train_cli


@pytest.fixture
def mock_remote():
    remote = MagicMock()
    remote.api = MagicMock()
    return remote


class TestRecreateTrainingJob:
    """Test cases for the recreate_training_job function."""

    def test_recreate_training_job_success(self, mock_remote):
        """Test recreating a training job with a specific job ID."""
        mock_remote.api.recreate_training_job.return_value = {
            "id": "test_job_124",
            "training_project": {"id": "project_456", "name": "aghilan-anime-project"},
        }

        mock_remote.api.search_training_jobs.return_value = [
            {
                "id": "test_job_123",
                "training_project": {
                    "id": "project_456",
                    "name": "aghilan-anime-project",
                },
            }
        ]

        result = train_cli.recreate_training_job(
            remote_provider=mock_remote, job_id="test_job_123"
        )

        assert result["id"] == "test_job_124"
        assert result["training_project"]["id"] == "project_456"
        assert result["training_project"]["name"] == "aghilan-anime-project"

        mock_remote.api.recreate_training_job.assert_called_once_with(
            "project_456", "test_job_123"
        )

    def test_recreate_training_job_no_job_found(self, mock_remote):
        """Test recreating a training job with a non-existent job ID."""
        mock_remote.api.search_training_jobs.return_value = []

        with pytest.raises(
            RuntimeError, match="No training job found with ID: nonexistent_job"
        ):
            train_cli.recreate_training_job(
                remote_provider=mock_remote, job_id="nonexistent_job"
            )

        mock_remote.api.search_training_jobs.assert_called_once_with(
            job_id="nonexistent_job"
        )
        mock_remote.api.recreate_training_job.assert_not_called()

    def test_recreate_training_job_without_job_id_success(self, mock_remote):
        """Test recreating a training job without specifying job ID (uses latest active)."""
        mock_remote.api.search_training_jobs.return_value = [
            {
                "id": "test_job_123",
                "training_project": {
                    "id": "project_456",
                    "name": "aghilan-anime-project",
                },
            }
        ]

        mock_remote.api.recreate_training_job.return_value = {
            "id": "test_job_124",
            "training_project": {"id": "project_456", "name": "aghilan-anime-project"},
        }

        # Mock the confirmation dialog
        with patch("truss.cli.train.core.inquirer.confirm") as mock_confirm:
            mock_confirm.return_value.execute.return_value = True

            result = train_cli.recreate_training_job(remote_provider=mock_remote)

            assert result["id"] == "test_job_124"
            assert result["training_project"]["id"] == "project_456"
            assert result["training_project"]["name"] == "aghilan-anime-project"

            # Verify the API calls
            mock_remote.api.search_training_jobs.assert_called_once_with(
                order_by=[{"field": "created_at", "order": "desc"}]
            )
            mock_remote.api.recreate_training_job.assert_called_once_with(
                "project_456", "test_job_123"
            )

    def test_recreate_training_job_without_job_id_no_active_jobs(self, mock_remote):
        """Test recreating a training job when no active jobs exist."""
        mock_remote.api.search_training_jobs.return_value = []

        # Should raise ClickException when no active jobs found
        with pytest.raises(
            click.ClickException,
            match="No training jobs found. Please start a job first.",
        ):
            train_cli.recreate_training_job(remote_provider=mock_remote)

    def test_recreate_training_job_without_job_id_user_cancels(self, mock_remote):
        """Test recreating a training job when user cancels the confirmation."""
        mock_remote.api.search_training_jobs.return_value = [
            {
                "id": "test_job_123",
                "training_project": {
                    "id": "project_456",
                    "name": "aghilan-anime-project",
                },
            }
        ]

        # Mock the confirmation dialog to return False
        with patch("truss.cli.train.core.inquirer.confirm") as mock_confirm:
            mock_confirm.return_value.execute.return_value = False

            # Should raise UsageError when user cancels
            with pytest.raises(click.UsageError, match="Training job not recreated"):
                train_cli.recreate_training_job(remote_provider=mock_remote)


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
            "project_456", "test_job_123", priority=42
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
