"""Migration commands for truss CLI.

Provides commands to migrate deprecated config formats to the new weights API.
"""

import difflib
import io
import os
import shutil
from pathlib import Path
from typing import Any

import rich_click as click
import yaml
from rich.console import Console
from rich.syntax import Syntax

from truss.base.constants import MODEL_CACHE_PATH
from truss.base.truss_config import ExternalDataItem, ModelRepo, ModelRepoSourceKind
from truss.cli.cli import truss_cli
from truss.cli.utils import common

console = Console()
error_console = Console(stderr=True)

# Data directory path (where external_data files are downloaded)
DATA_DIR_PATH = Path("/app/data")


def generate_source_uri(model: ModelRepo) -> str:
    """Generate source URI from ModelRepo based on its kind."""
    kind = model.kind
    repo_id = model.repo_id
    revision = model.revision

    if kind == ModelRepoSourceKind.HF:
        # HuggingFace: hf://owner/repo or hf://owner/repo@revision
        if revision:
            return f"hf://{repo_id}@{revision}"
        return f"hf://{repo_id}"
    elif kind == ModelRepoSourceKind.GCS:
        # GCS: repo_id should already have gs:// prefix or be a bucket path
        if repo_id.startswith("gs://"):
            return repo_id
        return f"gs://{repo_id}"
    elif kind == ModelRepoSourceKind.S3:
        # S3: repo_id should already have s3:// prefix or be a bucket path
        if repo_id.startswith("s3://"):
            return repo_id
        return f"s3://{repo_id}"
    elif kind == ModelRepoSourceKind.AZURE:
        # Azure: repo_id should already have azure:// prefix or be an account path
        if repo_id.startswith("azure://"):
            return repo_id
        return f"azure://{repo_id}"
    else:
        # Default to treating as HuggingFace
        if revision:
            return f"hf://{repo_id}@{revision}"
        return f"hf://{repo_id}"


def generate_mount_location_for_model(model: ModelRepo) -> str:
    """Generate mount_location for a ModelRepo.

    For v2 (use_volume=True): Uses volume_folder
    For v1 (use_volume=False): Generates from repo_id
    """
    if model.use_volume and model.volume_folder:
        # v2: Use the explicit volume_folder
        return str(MODEL_CACHE_PATH / model.volume_folder)

    # v1: Generate from repo_id
    kind = model.kind
    repo_id = model.repo_id

    if kind == ModelRepoSourceKind.HF:
        # Sanitize HuggingFace repo_id: owner/repo -> owner_repo
        sanitized = repo_id.replace("/", "_")
        return str(MODEL_CACHE_PATH / sanitized)
    elif kind in (
        ModelRepoSourceKind.GCS,
        ModelRepoSourceKind.S3,
        ModelRepoSourceKind.AZURE,
    ):
        # For cloud storage, extract bucket name from the path
        # Remove any scheme prefix first
        path = repo_id
        for prefix in ("gs://", "s3://", "azure://"):
            if path.startswith(prefix):
                path = path[len(prefix) :]
                break
        # Use the first path component (bucket name)
        bucket_name = path.split("/")[0]
        return str(MODEL_CACHE_PATH / bucket_name)
    else:
        # Default: sanitize repo_id
        sanitized = repo_id.replace("/", "_")
        return str(MODEL_CACHE_PATH / sanitized)


def convert_model_repo_to_weights(model: ModelRepo) -> dict:
    """Convert a ModelRepo to a WeightsSource dict."""
    source = generate_source_uri(model)
    mount_location = generate_mount_location_for_model(model)

    result: dict[str, Any] = {"source": source, "mount_location": mount_location}

    # Map runtime_secret_name to auth_secret_name
    if model.runtime_secret_name:
        result["auth_secret_name"] = model.runtime_secret_name

    # Preserve patterns if set
    if model.allow_patterns:
        result["allow_patterns"] = list(model.allow_patterns)
    if model.ignore_patterns:
        result["ignore_patterns"] = list(model.ignore_patterns)

    return result


def convert_external_data_to_weights(item: ExternalDataItem) -> dict:
    """Convert an ExternalDataItem to a WeightsSource dict."""
    # URL is already https://
    source = item.url

    # Mount location is /app/data/{local_data_path}
    mount_location = str(DATA_DIR_PATH / item.local_data_path)

    return {"source": source, "mount_location": mount_location}


def migrate_config_data(config_data: dict) -> tuple[list[dict], list[str]]:
    """Generate weights list from model_cache and external_data.

    Args:
        config_data: The raw config dictionary

    Returns:
        Tuple of (weights_list, warnings)
    """
    warnings = []
    weights_list = []

    # Migrate model_cache if present
    model_cache = config_data.get("model_cache", [])
    if model_cache:
        for model_dict in model_cache:
            # Parse as ModelRepo to get proper types
            model = ModelRepo.model_validate(dict(model_dict))

            # Warn about v1 HuggingFace requiring model.py changes
            if not model.use_volume and model.kind == ModelRepoSourceKind.HF:
                warnings.append(
                    f"v1 HuggingFace repo '{model.repo_id}' migrated. "
                    f"You may need to update model.py to use the new mount path: "
                    f"{generate_mount_location_for_model(model)}"
                )

            weights_list.append(convert_model_repo_to_weights(model))

    # Migrate external_data if present
    external_data = config_data.get("external_data")
    if external_data:
        for item_dict in external_data:
            item = ExternalDataItem.model_validate(dict(item_dict))
            weights_list.append(convert_external_data_to_weights(item))

    return weights_list, warnings


def dump_yaml_to_string(data) -> str:
    """Dump YAML data to string."""
    stream = io.StringIO()
    yaml.safe_dump(data, stream)
    return stream.getvalue()


def show_diff(original_yaml: str, migrated_yaml: str) -> None:
    """Display a colorized diff between original and migrated configs."""
    original_lines = original_yaml.splitlines(keepends=True)
    migrated_lines = migrated_yaml.splitlines(keepends=True)

    diff = difflib.unified_diff(
        original_lines,
        migrated_lines,
        fromfile="config.yaml (original)",
        tofile="config.yaml (migrated)",
        lineterm="",
    )

    diff_text = "".join(diff)
    if diff_text:
        syntax = Syntax(diff_text, "diff", theme="monokai", line_numbers=False)
        console.print(syntax)
    else:
        console.print("[yellow]No changes detected.[/yellow]")


@truss_cli.command()
@click.argument("target_directory", required=False, default=os.getcwd())
@common.common_options()
def migrate(target_directory: str) -> None:
    """Migrate model_cache and external_data to the new weights API.

    TARGET_DIRECTORY: A Truss directory. If none, use current directory.

    This command converts deprecated model_cache and external_data configurations
    to the unified weights API format. It shows a diff preview and asks for
    confirmation before applying changes.
    """
    target_path = Path(target_directory)
    config_path = target_path / "config.yaml"

    if not config_path.exists():
        error_console.print(f"[red]Error: No config.yaml found at {config_path}[/red]")
        raise SystemExit(1)

    # Read the original config
    with config_path.open() as f:
        original_yaml = f.read()

    with config_path.open() as f:
        config_data = yaml.safe_load(f)

    if config_data is None:
        config_data = {}

    # Check if already using weights - not an error, just nothing to do
    if config_data.get("weights"):
        console.print(
            "[yellow]Config already uses the weights API. Nothing to migrate.[/yellow]"
        )
        return

    # Check if there's anything to migrate
    has_model_cache = bool(config_data.get("model_cache"))
    has_external_data = bool(config_data.get("external_data"))

    if not has_model_cache and not has_external_data:
        console.print(
            "[yellow]No model_cache or external_data found. Nothing to migrate.[/yellow]"
        )
        return

    # Generate weights from model_cache and external_data
    weights_list, warnings = migrate_config_data(config_data)

    # Modify the config in place
    if "model_cache" in config_data:
        del config_data["model_cache"]
    if "external_data" in config_data:
        del config_data["external_data"]

    # Add weights
    config_data["weights"] = weights_list

    # Generate migrated YAML string
    migrated_yaml = dump_yaml_to_string(config_data)

    # Show warnings
    for warning in warnings:
        console.print(f"[yellow]Warning:[/yellow] {warning}")

    # Show diff
    console.print("\n[bold]Proposed changes:[/bold]\n")
    show_diff(original_yaml, migrated_yaml)

    # Prompt for confirmation
    console.print()
    if not click.confirm("Apply these changes?", default=False):
        console.print("[yellow]Migration cancelled.[/yellow]")
        return

    # Create backup
    backup_path = config_path.with_suffix(".yaml.bak")
    shutil.copy(config_path, backup_path)
    console.print(f"[dim]Backup created: {backup_path}[/dim]")

    # Write migrated config
    with config_path.open("w") as f:
        yaml.safe_dump(config_data, f)

    console.print("[green]Migration complete![/green]")


# For backwards compatibility with tests that import migrate_config
def migrate_config(config_dict: dict) -> tuple[dict, list[str]]:
    """Migrate model_cache and external_data to weights.

    Args:
        config_dict: The raw config dictionary

    Returns:
        Tuple of (migrated_config_dict, warnings)
    """
    weights_list, warnings = migrate_config_data(config_dict)

    migrated = dict(config_dict)

    # Remove old keys
    if "model_cache" in migrated:
        del migrated["model_cache"]
    if "external_data" in migrated:
        del migrated["external_data"]

    # Add weights if we have any
    if weights_list:
        migrated["weights"] = weights_list

    return migrated, warnings
