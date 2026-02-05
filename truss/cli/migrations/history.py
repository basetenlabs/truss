"""Migration history tracking.

Tracks which migrations have been applied to each Truss directory.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import truss


def get_migration_history_path(truss_dir: Path) -> Path:
    """Get the path to the migration history file for a Truss directory."""
    truss_meta_dir = truss_dir / ".truss"
    truss_meta_dir.mkdir(exist_ok=True)
    return truss_meta_dir / "migration_history.json"


def load_migration_history(truss_dir: Path) -> Dict[str, Any]:
    """Load migration history from .truss/migration_history.json."""
    history_path = get_migration_history_path(truss_dir)
    if not history_path.exists():
        return {"applied_migrations": {}}

    try:
        with history_path.open() as f:
            history = json.load(f)
        # Ensure applied_migrations exists
        if "applied_migrations" not in history:
            history["applied_migrations"] = {}
        return history
    except (json.JSONDecodeError, IOError):
        # If file is corrupted or can't be read, return empty history
        return {"applied_migrations": {}}


def save_migration_history(truss_dir: Path, history: Dict[str, Any]) -> None:
    """Save migration history to .truss/migration_history.json."""
    history_path = get_migration_history_path(truss_dir)
    with history_path.open("w") as f:
        json.dump(history, f, indent=2)


def is_migration_applied(truss_dir: Path, migration_id: str) -> bool:
    """Check if a migration has been applied."""
    history = load_migration_history(truss_dir)
    return migration_id in history.get("applied_migrations", {})


def record_migration_applied(
    truss_dir: Path, migration_id: str, backup_file: str
) -> None:
    """Record that a migration has been applied."""
    history = load_migration_history(truss_dir)
    history["applied_migrations"][migration_id] = {
        "applied_at": datetime.now().isoformat(),
        "applied_by_version": truss.__version__,
        "backup_file": backup_file,
    }
    save_migration_history(truss_dir, history)
