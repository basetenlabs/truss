from unittest.mock import MagicMock, patch

import pytest

from truss.cli.train import core as train_cli


@pytest.fixture
def mock_remote():
    remote = MagicMock()
    remote.api = MagicMock()
    return remote


@pytest.fixture
def mock_job_response():
    return [
        {
            "id": "test_job_123",
            "training_project": {"id": "project_456", "name": "test_project"},
        }
    ]


def test_download_training_job_success(tmp_path, mock_remote, mock_job_response):
    # Setup
    mock_remote.api.search_training_jobs.return_value = mock_job_response
    mock_remote.api.get_training_job_presigned_url.return_value = "https://test-url.com"
    mock_remote.api.get_from_presigned_url.return_value = b"test_content"

    # Execute
    result = train_cli.download_training_job_data(
        remote_provider=mock_remote,
        job_id="test_job_123",
        target_directory=str(tmp_path),
    )

    # Assert
    assert result.exists()
    assert result.name == "test_project_test_job_123.tgz"
    assert result.read_bytes() == b"test_content"

    # Verify API calls
    mock_remote.api.search_training_jobs.assert_called_once_with(job_id="test_job_123")
    mock_remote.api.get_training_job_presigned_url.assert_called_once_with(
        project_id="project_456", job_id="test_job_123"
    )
    mock_remote.api.get_from_presigned_url.assert_called_once_with(
        "https://test-url.com"
    )


def test_download_training_job_no_job_found(mock_remote):
    # Setup
    mock_remote.api.search_training_jobs.return_value = []

    # Execute and Assert
    with pytest.raises(
        RuntimeError, match="No training job found with ID: nonexistent_job"
    ):
        train_cli.download_training_job_data(
            remote_provider=mock_remote, job_id="nonexistent_job", target_directory=None
        )


def test_download_training_job_default_directory(
    tmp_path, mock_remote, mock_job_response
):
    # Setup
    mock_remote.api.search_training_jobs.return_value = mock_job_response
    mock_remote.api.get_training_job_presigned_url.return_value = "https://test-url.com"
    mock_remote.api.get_from_presigned_url.return_value = b"test_content"

    test_dir = tmp_path / "current/dir"
    test_dir.mkdir(parents=True)

    # Execute
    with patch("pathlib.Path.cwd") as mock_cwd:
        mock_cwd.return_value = test_dir
        result = train_cli.download_training_job_data(
            remote_provider=mock_remote, job_id="test_job_123", target_directory=None
        )

    # Assert
    expected_path = test_dir / "test_project_test_job_123.tgz"
    assert result.resolve() == expected_path.resolve()


@pytest.mark.parametrize(
    "target_dir", ["test/path", "relative/path", "single_dir", None]
)
def test_download_training_job_different_directories(
    target_dir, mock_remote, mock_job_response, tmp_path
):
    # Setup
    mock_remote.api.search_training_jobs.return_value = mock_job_response
    mock_remote.api.get_training_job_presigned_url.return_value = "https://test-url.com"
    mock_remote.api.get_from_presigned_url.return_value = b"test_content"

    if target_dir:
        full_path = tmp_path / target_dir
        full_path.mkdir(parents=True, exist_ok=True)
    else:
        full_path = tmp_path

    # Execute
    with patch("pathlib.Path.cwd") as mock_cwd:
        mock_cwd.return_value = full_path
        result = train_cli.download_training_job_data(
            remote_provider=mock_remote,
            job_id="test_job_123",
            target_directory=str(full_path) if target_dir else None,
        )

    # Assert
    expected_path = full_path / "test_project_test_job_123.tgz"
    assert result.resolve() == expected_path.resolve()
