"""Tests for migration registry."""

import pytest

from truss.cli.migrations.registry import (
    ConfigMigration,
    _check_model_cache_to_weights_needed,
    get_migrations,
)


@pytest.mark.parametrize(
    "config,expected",
    [
        ({"model_cache": [{"repo_id": "test/model"}]}, True),
        ({"external_data": [{"url": "https://example.com"}]}, True),
        (
            {
                "model_cache": [{"repo_id": "test/model"}],
                "external_data": [{"url": "https://example.com"}],
            },
            True,
        ),
        ({"weights": [{"source": "hf://test/model"}]}, False),
        (
            {
                "model_cache": [{"repo_id": "test/model"}],
                "weights": [{"source": "hf://test/model"}],
            },
            False,
        ),
        ({}, False),
        ({"model_name": "test", "python_version": "py311"}, False),
    ],
)
def test_check_model_cache_to_weights_needed(config, expected):
    """Test _check_model_cache_to_weights_needed with various configs."""
    assert _check_model_cache_to_weights_needed(config) == expected


class TestMigrationRegistry:
    """Tests for migration registry."""

    @pytest.fixture
    def migrations(self):
        """Get migrations list."""
        return get_migrations()

    def test_registry_has_migrations(self, migrations):
        """Test that registry contains migrations."""
        assert len(migrations) > 0

    def test_model_cache_migration_registered(self, migrations):
        """Test that model_cache_to_weights migration is registered."""
        assert "model_cache_to_weights_v0.12.0" in [m.id for m in migrations]

    def test_migration_has_required_fields(self, migrations):
        """Test that all migrations have required fields."""
        for migration in migrations:
            assert isinstance(migration, ConfigMigration)
            assert (
                migration.id
                and migration.introduced_in_version
                and migration.description
            )
            assert callable(migration.check_function) and callable(
                migration.apply_function
            )

    def test_model_cache_migration_details(self, migrations):
        """Test details of model_cache_to_weights migration."""
        migration = next(
            m for m in migrations if m.id == "model_cache_to_weights_v0.12.0"
        )
        assert migration.introduced_in_version == "0.12.0"
        assert migration.required_version == "0.12.0"
        assert "model_cache" in migration.description.lower()
        assert "weights" in migration.description.lower()
