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
                statuses=train_cli.ACTIVE_JOB_STATUSES,
                order_by=[{"field": "created_at", "order": "desc"}],
            )
            mock_remote.api.recreate_training_job.assert_called_once_with(
                "project_456", "test_job_123"
            )

    def test_recreate_training_job_without_job_id_no_active_jobs(self, mock_remote):
        """Test recreating a training job when no active jobs exist."""
        mock_remote.api.search_training_jobs.return_value = []

        # Should raise ClickException when no active jobs found
        with pytest.raises(click.ClickException, match="No active training jobs found"):
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
