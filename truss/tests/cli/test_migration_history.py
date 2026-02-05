"""Tests for migration history tracking."""

import json

import pytest

from truss.cli.migrations.history import (
    get_migration_history_path,
    is_migration_applied,
    load_migration_history,
    record_migration_applied,
    save_migration_history,
)


@pytest.fixture
def truss_dir(tmp_path):
    """Create a temporary Truss directory."""
    truss_dir = tmp_path / "my_truss"
    truss_dir.mkdir()
    return truss_dir


class TestMigrationHistoryPath:
    """Tests for get_migration_history_path."""

    def test_creates_truss_dir(self, truss_dir):
        """Test that .truss directory is created if it doesn't exist."""
        history_path = get_migration_history_path(truss_dir)
        assert history_path.parent.exists()
        assert history_path.parent.name == ".truss"
        assert history_path.name == "migration_history.json"

    def test_uses_existing_truss_dir(self, truss_dir):
        """Test that existing .truss directory is used."""
        truss_meta_dir = truss_dir / ".truss"
        truss_meta_dir.mkdir()

        history_path = get_migration_history_path(truss_dir)
        assert history_path.parent == truss_meta_dir


class TestLoadMigrationHistory:
    """Tests for load_migration_history."""

    def test_loads_existing_history(self, truss_dir):
        """Test loading existing migration history."""
        history_path = get_migration_history_path(truss_dir)
        history_data = {
            "applied_migrations": {
                "migration_1": {
                    "applied_at": "2024-01-15T10:30:00Z",
                    "applied_by_version": "0.12.0",
                    "backup_file": "config.yaml.bak.20240115-103000",
                }
            }
        }

        with history_path.open("w") as f:
            json.dump(history_data, f)

        assert load_migration_history(truss_dir) == history_data

    def test_returns_empty_if_no_file(self, truss_dir):
        """Test that missing history file returns empty structure."""
        assert load_migration_history(truss_dir) == {"applied_migrations": {}}

    def test_handles_corrupted_json(self, truss_dir):
        """Test that corrupted JSON returns empty history."""
        history_path = get_migration_history_path(truss_dir)
        with history_path.open("w") as f:
            f.write("{ invalid json }")

        assert load_migration_history(truss_dir) == {"applied_migrations": {}}

    def test_handles_missing_applied_migrations_key(self, truss_dir):
        """Test that missing applied_migrations key is handled."""
        history_path = get_migration_history_path(truss_dir)
        with history_path.open("w") as f:
            json.dump({"other_key": "value"}, f)

        history = load_migration_history(truss_dir)
        assert history["applied_migrations"] == {}


class TestSaveMigrationHistory:
    """Tests for save_migration_history."""

    def test_saves_history(self, truss_dir):
        """Test saving migration history."""
        history = {
            "applied_migrations": {
                "migration_1": {
                    "applied_at": "2024-01-15T10:30:00Z",
                    "applied_by_version": "0.12.0",
                    "backup_file": "config.yaml.bak",
                }
            }
        }

        save_migration_history(truss_dir, history)

        history_path = get_migration_history_path(truss_dir)
        assert history_path.exists()
        with history_path.open() as f:
            assert json.load(f) == history


class TestIsMigrationApplied:
    """Tests for is_migration_applied."""

    @pytest.fixture
    def history_with_migration(self, truss_dir):
        """Create history file with a migration."""
        history_path = get_migration_history_path(truss_dir)
        history_data = {
            "applied_migrations": {
                "migration_1": {
                    "applied_at": "2024-01-15T10:30:00Z",
                    "applied_by_version": "0.12.0",
                    "backup_file": "config.yaml.bak",
                }
            }
        }
        with history_path.open("w") as f:
            json.dump(history_data, f)
        return truss_dir

    @pytest.mark.parametrize(
        "migration_id,expected",
        [("migration_1", True), ("migration_2", False), ("nonexistent", False)],
    )
    def test_checks_migration_status(
        self, history_with_migration, migration_id, expected
    ):
        """Test checking if migration is applied."""
        assert is_migration_applied(history_with_migration, migration_id) == expected

    def test_returns_false_for_empty_history(self, truss_dir):
        """Test that empty history returns False."""
        assert is_migration_applied(truss_dir, "any_migration") is False


class TestRecordMigrationApplied:
    """Tests for record_migration_applied."""

    def test_records_migration(self, truss_dir):
        """Test recording a migration."""
        record_migration_applied(
            truss_dir, "migration_1", "config.yaml.bak.20240115-103000"
        )

        history = load_migration_history(truss_dir)
        migration_info = history["applied_migrations"]["migration_1"]
        assert "applied_at" in migration_info
        assert "applied_by_version" in migration_info
        assert migration_info["backup_file"] == "config.yaml.bak.20240115-103000"

    def test_appends_to_existing_history(self, truss_dir):
        """Test that recording doesn't overwrite existing migrations."""
        record_migration_applied(truss_dir, "migration_1", "backup1.bak")
        record_migration_applied(truss_dir, "migration_2", "backup2.bak")

        history = load_migration_history(truss_dir)
        assert len(history["applied_migrations"]) == 2
        assert set(history["applied_migrations"].keys()) == {
            "migration_1",
            "migration_2",
        }
