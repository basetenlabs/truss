"""Tests for fetch_project_by_name_or_id function with team filtering."""

from unittest.mock import Mock, patch

import click
import pytest

from truss.cli.train.core import fetch_project_by_name_or_id
from truss.remote.baseten.remote import BasetenRemote


class TestFetchProjectByNameOrId:
    """Test fetch_project_by_name_or_id function."""

    @staticmethod
    def _setup_mock_remote(projects):
        mock_remote = Mock(spec=BasetenRemote)
        mock_api = Mock()
        mock_remote.api = mock_api
        mock_api.list_training_projects.return_value = projects
        return mock_remote

    def test_lookup_by_id_and_unique_name(self):
        """Test lookup by project ID and unique project name."""
        projects = [
            {"id": "proj-123", "name": "my-project", "team_name": "Team Alpha"},
            {"id": "proj-456", "name": "other-project", "team_name": "Team Beta"},
        ]
        mock_remote = self._setup_mock_remote(projects)

        # Lookup by ID
        result = fetch_project_by_name_or_id(mock_remote, "proj-123")
        assert result["id"] == "proj-123"

        # Lookup by unique name
        result = fetch_project_by_name_or_id(mock_remote, "my-project")
        assert result["id"] == "proj-123"

        # ID lookup ignores team filter
        result = fetch_project_by_name_or_id(
            mock_remote, "proj-123", team_name="NonExistent"
        )
        assert result["id"] == "proj-123"

    def test_team_filter_and_ambiguous_names(self):
        """Test --team filtering and prompting for ambiguous project names."""
        projects = [
            {"id": "proj-123", "name": "my-project", "team_name": "Team Alpha"},
            {"id": "proj-456", "name": "my-project", "team_name": "Team Beta"},
            {"id": "proj-789", "name": "my-project", "team_name": "Team Gamma"},
        ]
        mock_remote = self._setup_mock_remote(projects)

        # --team filters correctly
        result = fetch_project_by_name_or_id(
            mock_remote, "my-project", team_name="Team Beta"
        )
        assert result["id"] == "proj-456"

        # Ambiguous name with --team does NOT prompt
        with patch("truss.cli.train.core.inquirer.select") as mock_select:
            result = fetch_project_by_name_or_id(
                mock_remote, "my-project", team_name="Team Alpha"
            )
            mock_select.assert_not_called()
            assert result["id"] == "proj-123"

        # Ambiguous name without --team prompts user (only once)
        with patch("truss.cli.train.core.inquirer.select") as mock_select:
            mock_select.return_value.execute.return_value = "Team Gamma"
            result = fetch_project_by_name_or_id(mock_remote, "my-project")
            assert mock_select.call_count == 1
            assert "Team Alpha" in mock_select.call_args[1]["choices"]
            assert "Team Beta" in mock_select.call_args[1]["choices"]
            assert "Team Gamma" in mock_select.call_args[1]["choices"]
            assert result["id"] == "proj-789"

    def test_error_cases(self):
        """Test error cases for project lookup."""
        projects = [{"id": "proj-123", "name": "my-project", "team_name": "Team Alpha"}]
        mock_remote = self._setup_mock_remote(projects)

        # Project not found
        with pytest.raises(click.ClickException) as exc_info:
            fetch_project_by_name_or_id(mock_remote, "nonexistent")
        assert "not found" in str(exc_info.value)

        # Project not found in specified team
        with pytest.raises(click.ClickException) as exc_info:
            fetch_project_by_name_or_id(
                mock_remote, "my-project", team_name="Team Beta"
            )
        assert "not found" in str(exc_info.value)
        assert "Team Beta" in str(exc_info.value)

        # Empty project list
        empty_remote = self._setup_mock_remote([])
        with pytest.raises(click.ClickException) as exc_info:
            fetch_project_by_name_or_id(empty_remote, "my-project")
        assert "not found" in str(exc_info.value)
