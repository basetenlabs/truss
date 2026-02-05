"""Migration registry for truss config migrations.

Migrations are registered here and bundled with each truss release.
When a new truss version is released with a migration, users who upgrade
will have access to that migration.
"""

from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class ConfigMigration:
    """Represents a config migration that can be applied."""

    id: str  # Unique identifier, e.g., "model_cache_to_weights_v0.12.0"
    introduced_in_version: (
        str  # Version when this migration became available, e.g., "0.12.0"
    )
    description: str
    check_function: Callable[[dict], bool]  # Returns True if migration is needed
    apply_function: Callable[
        [dict], tuple[dict, list[str]]
    ]  # Returns (migrated_config, warnings)
    required_version: Optional[str] = None  # Minimum truss version required to apply


def _check_model_cache_to_weights_needed(config: dict) -> bool:
    """Check if model_cache_to_weights migration is needed."""
    has_model_cache = bool(config.get("model_cache"))
    has_external_data = bool(config.get("external_data"))
    has_weights = bool(config.get("weights"))
    return (has_model_cache or has_external_data) and not has_weights


def _get_migrations():
    """Get migrations list.

    Note: migrate_config is imported here to avoid circular dependency:
    registry -> migrate_commands -> detection -> registry
    """
    from truss.cli.migrate_commands import migrate_config

    return [
        ConfigMigration(
            id="model_cache_to_weights_v0.12.0",
            introduced_in_version="0.12.0",
            description="Migrate model_cache and external_data to weights API",
            check_function=_check_model_cache_to_weights_needed,
            apply_function=migrate_config,
            required_version="0.12.0",
        )
    ]


# Lazy evaluation to avoid circular imports
MIGRATIONS = None


def get_migrations():
    """Get the migrations list, initializing it if needed."""
    global MIGRATIONS
    if MIGRATIONS is None:
        MIGRATIONS = _get_migrations()
    return MIGRATIONS
