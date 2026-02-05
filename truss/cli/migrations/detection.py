"""Migration detection logic.

Detects which migrations are available and applicable for a given Truss directory.
"""

from pathlib import Path
from typing import Any, Dict, Optional

from packaging import version
from ruamel.yaml import YAML

import truss
from truss.cli.migrations.history import is_migration_applied
from truss.cli.migrations.registry import ConfigMigration, get_migrations


def get_available_migrations(
    truss_dir: Path, config_data: Optional[Dict[str, Any]] = None
) -> list[ConfigMigration]:
    """Get list of migrations that are available and applicable.

    Args:
        truss_dir: Path to the Truss directory
        config_data: Optional config dict. If not provided, will be loaded from config.yaml

    Returns:
        List of applicable migrations
    """
    current_version = truss.__version__

    applicable_migrations = []
    migrations = get_migrations()

    for migration in migrations:
        # Filter by version: only show migrations where current_version >= introduced_in_version
        try:
            if version.parse(current_version) < version.parse(
                migration.introduced_in_version
            ):
                continue
        except version.InvalidVersion:
            # If version parsing fails, skip this migration
            continue

        # Filter by history: skip migrations already applied
        if is_migration_applied(truss_dir, migration.id):
            continue

        # Filter by need: only show if migration is actually needed
        if config_data is None:
            config_path = truss_dir / "config.yaml"
            if config_path.exists():
                yaml = YAML()
                with config_path.open() as f:
                    config_data = yaml.load(f) or {}
            else:
                config_data = {}

        if not migration.check_function(config_data):
            continue

        applicable_migrations.append(migration)

    return applicable_migrations
