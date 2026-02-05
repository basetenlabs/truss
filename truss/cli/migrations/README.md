# Truss Config Migrations

A generic system for migrating Truss configuration files to newer formats as the project evolves.

## HOWTO

### For Users

#### Running Migrations

When you upgrade `truss`, you may be notified about available config migrations. To apply them:

```bash
truss migrate
```

This command will:
1. Detect which migrations are applicable to your config
2. Show you a preview of the changes
3. Ask for confirmation before applying
4. Create a backup of your `config.yaml` before making changes

#### Example

```bash
$ truss migrate
Found 1 applicable migration(s):

📦 model_cache_to_weights_v0.12.0
   Migrate model_cache and external_data to weights API

Proposed changes:
[Shows diff of changes]

Apply migration 'model_cache_to_weights_v0.12.0'? [y/N]: y
Backup created: config.yaml.bak.20240115-103000
Migration 'model_cache_to_weights_v0.12.0' complete!
```

#### Migration History

Migrations are tracked in `.truss/migration_history.json`. This ensures:
- Migrations are only applied once
- You can see which migrations have been applied
- Future features like rollback can use this history

#### Safety Features

- **Backups**: Every migration creates a timestamped backup
- **Preview**: See changes before applying
- **Idempotent**: Migrations won't run twice on the same config
- **Version-gated**: Only migrations compatible with your `truss` version are shown

## Design

### Architecture

The migration system consists of three main components:

1. **Registry** (`registry.py`): Defines available migrations
2. **Detection** (`detection.py`): Determines which migrations are applicable
3. **History** (`history.py`): Tracks applied migrations

### Adding a New Migration

When introducing a breaking config change in a new `truss` version:

1. **Define the migration in `registry.py`**:

```python
def _check_my_migration_needed(config: dict) -> bool:
    """Check if migration is needed."""
    return "old_field" in config and "new_field" not in config

def _apply_my_migration(config: dict) -> tuple[dict, list[str]]:
    """Apply the migration."""
    migrated = config.copy()
    migrated["new_field"] = migrated.pop("old_field")
    return migrated, []

def _get_migrations():
    from truss.cli.migrate_commands import migrate_config

    return [
        # ... existing migrations ...
        ConfigMigration(
            id="old_field_to_new_field_v0.13.0",
            introduced_in_version="0.13.0",
            description="Migrate old_field to new_field",
            check_function=_check_my_migration_needed,
            apply_function=_apply_my_migration,
            required_version="0.13.0",
        ),
    ]
```

2. **Migration ID Format**: Use `{description}_v{version}` (e.g., `model_cache_to_weights_v0.12.0`)

3. **Version Gating**: Set `introduced_in_version` to the release where the migration becomes available

4. **Check Function**: Should return `True` only when migration is actually needed

5. **Apply Function**: Returns `(migrated_config, warnings)` tuple

### Migration Lifecycle

1. **User upgrades `truss`** → New migrations become available
2. **User runs `truss migrate`** → System detects applicable migrations
3. **System filters migrations** by:
   - Version compatibility (current >= introduced_in_version)
   - History (not already applied)
   - Need (check_function returns True)
4. **User confirms** → Migration applied, history updated

### Migration History Format

```json
{
  "applied_migrations": {
    "model_cache_to_weights_v0.12.0": {
      "applied_at": "2024-01-15T10:30:00Z",
      "applied_by_version": "0.12.3",
      "backup_file": "config.yaml.bak.20240115-103000"
    }
  }
}
```

### Best Practices

- **Idempotent**: Migrations should be safe to run multiple times (though they won't be)
- **Reversible**: Consider how users might rollback (backups help)
- **Clear descriptions**: Users need to understand what the migration does
- **Warnings**: Use the warnings list to notify users about manual steps needed
- **Test thoroughly**: Migrations modify user configs - test edge cases

### Testing

See `truss/tests/cli/test_migration_*.py` for test examples:
- `test_migration_history.py`: History tracking
- `test_migration_registry.py`: Registry and check functions
- `test_migration_detection.py`: Detection logic
- `test_migrate_with_registry.py`: CLI integration
