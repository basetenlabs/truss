"""Tests for migrate command using the registry system."""

from unittest.mock import patch

import pytest
import yaml as pyyaml
from click.testing import CliRunner

from truss.cli.cli import truss_cli
from truss.cli.migrations.history import (
    load_migration_history,
    record_migration_applied,
)


class TestMigrateWithRegistry:
    """Tests for migrate command using registry system."""

    @pytest.fixture
    def truss_dir(self, tmp_path):
        """Create a temporary Truss directory with config."""
        truss_dir = tmp_path / "my_truss"
        truss_dir.mkdir()
        config_path = truss_dir / "config.yaml"
        config_path.write_text(
            """
model_name: test
model_cache:
  - repo_id: meta-llama/Llama-2-7b
    revision: main
    use_volume: true
    volume_folder: llama-7b
"""
        )
        return truss_dir

    @pytest.fixture
    def runner(self):
        """Create a CLI runner."""
        return CliRunner()

    def test_migrate_detects_available_migrations(self, truss_dir, runner):
        """Test that migrate command detects available migrations."""
        with patch("click.confirm", return_value=False):
            result = runner.invoke(truss_cli, ["migrate", str(truss_dir)], input="n\n")

            # Should show migration is available
            assert (
                "applicable migration" in result.output.lower()
                or "migration" in result.output.lower()
            )
            assert result.exit_code == 0

    def test_migrate_applies_migration(self, truss_dir, runner):
        """Test that migrate command applies migration when confirmed."""
        runner.invoke(
            truss_cli,
            ["migrate", str(truss_dir)],
            input="y\n",  # Confirm migration
        )

        # Check that config was migrated
        config_path = truss_dir / "config.yaml"
        with config_path.open() as f:
            config_content = f.read()

        assert "weights" in config_content
        with config_path.open() as f:
            assert "model_cache" not in pyyaml.safe_load(f)

        # Check that migration was recorded
        history = load_migration_history(truss_dir)
        assert "model_cache_to_weights_v0.12.0" in history["applied_migrations"]

    def test_migrate_creates_backup(self, truss_dir, runner):
        """Test that migrate command creates backup file."""
        config_path = truss_dir / "config.yaml"
        original_content = config_path.read_text()

        runner.invoke(
            truss_cli,
            ["migrate", str(truss_dir)],
            input="y\n",  # Confirm migration
        )

        # Check backup was created
        backup_files = list(truss_dir.glob("config.yaml.bak.*"))
        assert len(backup_files) > 0

        # Check backup content matches original
        backup_content = backup_files[0].read_text()
        assert backup_content == original_content

    def test_migrate_shows_no_migrations_when_none_applicable(self, tmp_path, runner):
        """Test that migrate shows message when no migrations are applicable."""
        truss_dir = tmp_path / "my_truss"
        truss_dir.mkdir()
        config_path = truss_dir / "config.yaml"
        config_path.write_text(
            """
model_name: test
weights:
  - source: hf://test/model
"""
        )

        result = runner.invoke(truss_cli, ["migrate", str(truss_dir)])

        assert (
            "no applicable migrations" in result.output.lower()
            or "up to date" in result.output.lower()
        )
        assert result.exit_code == 0

    def test_migrate_skips_already_applied_migrations(self, truss_dir, runner):
        """Test that already-applied migrations are skipped."""
        record_migration_applied(
            truss_dir, "model_cache_to_weights_v0.12.0", "backup.bak"
        )
        result = runner.invoke(truss_cli, ["migrate", str(truss_dir)])
        assert (
            "no applicable migrations" in result.output.lower()
            or "up to date" in result.output.lower()
        )

    def test_migrate_handles_multiple_migrations(self, truss_dir, runner):
        """Test that migrate handles multiple applicable migrations."""
        # This test will be more relevant when we have multiple migrations
        # For now, just verify it works with one
        result = runner.invoke(
            truss_cli,
            ["migrate", str(truss_dir)],
            input="y\n",  # Confirm migration
        )

        assert result.exit_code == 0

    def test_migrate_preserves_other_config_fields(self, truss_dir, runner):
        """Test that migrate preserves other config fields."""
        config_path = truss_dir / "config.yaml"
        config_path.write_text(
            """
model_name: test
python_version: py311
requirements:
  - torch
  - transformers
model_cache:
  - repo_id: test/model
    use_volume: true
    volume_folder: model
"""
        )

        with patch("click.confirm", return_value=True):
            runner.invoke(truss_cli, ["migrate", str(truss_dir)])

            # Check other fields are preserved
            with config_path.open() as f:
                config_content = f.read()

            assert "model_name: test" in config_content
            assert "python_version: py311" in config_content
            assert "requirements" in config_content
