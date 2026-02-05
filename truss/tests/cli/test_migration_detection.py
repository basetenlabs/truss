"""Tests for migration detection logic."""

from unittest.mock import patch

import pytest

from truss.cli.migrations.detection import get_available_migrations
from truss.cli.migrations.history import record_migration_applied
from truss.cli.migrations.registry import get_migrations


class TestGetAvailableMigrations:
    """Tests for get_available_migrations function."""

    @pytest.fixture
    def truss_dir(self, tmp_path):
        """Create a temporary Truss directory."""
        truss_dir = tmp_path / "my_truss"
        truss_dir.mkdir()
        config_path = truss_dir / "config.yaml"
        config_path.write_text("model_name: test\n")
        return truss_dir

    @pytest.mark.parametrize(
        "current_version,introduced_version,should_show",
        [
            ("0.12.0", "0.12.0", True),  # Same version
            ("0.13.0", "0.12.0", True),  # Newer version
            ("0.11.0", "0.12.0", False),  # Older version
            ("0.12.1", "0.12.0", True),  # Patch version newer
            ("0.12.0", "0.12.1", False),  # Patch version older
        ],
    )
    def test_version_filtering(
        self, truss_dir, current_version, introduced_version, should_show
    ):
        """Test that migrations are filtered by version."""
        config_data = {"model_cache": [{"repo_id": "test/model"}]}

        with patch("truss.cli.migrations.detection.truss.__version__", current_version):
            # Mock the migration to have the test version
            migrations = get_migrations()
            with patch.object(
                migrations[0], "introduced_in_version", introduced_version
            ):
                migrations = get_available_migrations(truss_dir, config_data)

                if should_show:
                    assert len(migrations) > 0
                else:
                    assert len(migrations) == 0

    def test_filters_by_history(self, truss_dir):
        """Test that already-applied migrations are filtered out."""
        config_data = {"model_cache": [{"repo_id": "test/model"}]}

        # Get migrations before applying
        migrations_before = get_available_migrations(truss_dir, config_data)
        assert len(migrations_before) > 0

        # Record a migration as applied
        migration_id = migrations_before[0].id
        record_migration_applied(truss_dir, migration_id, "backup.bak")

        # Get migrations after applying
        migrations_after = get_available_migrations(truss_dir, config_data)
        assert len(migrations_after) == 0

    @pytest.mark.parametrize(
        "config_data,should_show",
        [
            # Needs migration
            ({"model_cache": [{"repo_id": "test/model"}]}, True),
            ({"external_data": [{"url": "https://example.com"}]}, True),
            # Doesn't need migration
            ({"weights": [{"source": "hf://test/model"}]}, False),
            ({}, False),
            ({"model_name": "test"}, False),
        ],
    )
    def test_filters_by_need(self, truss_dir, config_data, should_show):
        """Test that migrations are filtered by whether they're actually needed."""
        migrations = get_available_migrations(truss_dir, config_data)

        if should_show:
            assert len(migrations) > 0
        else:
            assert len(migrations) == 0

    def test_loads_config_if_not_provided(self, truss_dir):
        """Test that config is loaded from file if not provided."""
        # Write config with model_cache
        config_path = truss_dir / "config.yaml"
        config_path.write_text(
            """
model_name: test
model_cache:
  - repo_id: test/model
"""
        )

        # Don't provide config_data
        migrations = get_available_migrations(truss_dir, config_data=None)

        assert len(migrations) > 0

    def test_handles_missing_config_file(self, tmp_path):
        """Test that missing config file is handled gracefully."""
        truss_dir = tmp_path / "my_truss"
        truss_dir.mkdir()
        # No config.yaml

        migrations = get_available_migrations(truss_dir, config_data=None)

        # Should return empty list since no migration is needed
        assert len(migrations) == 0

    def test_handles_invalid_version(self, truss_dir):
        """Test that invalid version strings are handled."""
        config_data = {"model_cache": [{"repo_id": "test/model"}]}

        with patch(
            "truss.cli.migrations.detection.truss.__version__", "invalid-version"
        ):
            # Should not crash, just skip migrations with version issues
            migrations = get_available_migrations(truss_dir, config_data)
            # Result depends on how version parsing fails, but shouldn't crash
            assert isinstance(migrations, list)

    def test_returns_all_applicable_migrations(self, truss_dir):
        """Test that applicable migrations are returned."""
        migrations = get_available_migrations(
            truss_dir, {"model_cache": [{"repo_id": "test/model"}]}
        )
        assert len(migrations) >= 1
        assert all(m.id == "model_cache_to_weights_v0.12.0" for m in migrations)
