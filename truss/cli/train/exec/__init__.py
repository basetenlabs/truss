"""Run a local directory as a Baseten training job.

The generic layer under `truss loops exec`: it assembles the job and knows nothing
about what the command being run is for. Callers supply every policy decision --
compute shape, cache volume, credentials -- so a second caller can choose
differently.
"""

from .builder import (
    DEFAULT_EXEC_PROJECT_NAME,
    PYTHON_BASE_IMAGE,
    SUPPORTED_EXEC_ACCELERATORS,
    build_exec_project,
    build_start_commands,
    default_base_image,
    resolve_workspace_root,
    validate_workspace_root,
)
from .project import Project, get_project_type
from .secrets import (
    API_KEYS_SETTINGS_URL,
    BASETEN_API_KEY_ENV_VAR,
    SECRETS_SETTINGS_URL,
    OrphanedApiKeyError,
    ensure_team_api_key_secret,
    parse_environment_variables,
    team_api_key_secret_name,
    validate_secret_references,
)
from .uv import UvProject

__all__ = [
    "API_KEYS_SETTINGS_URL",
    "BASETEN_API_KEY_ENV_VAR",
    "DEFAULT_EXEC_PROJECT_NAME",
    "PYTHON_BASE_IMAGE",
    "SECRETS_SETTINGS_URL",
    "SUPPORTED_EXEC_ACCELERATORS",
    "OrphanedApiKeyError",
    "Project",
    "UvProject",
    "build_exec_project",
    "build_start_commands",
    "default_base_image",
    "ensure_team_api_key_secret",
    "get_project_type",
    "parse_environment_variables",
    "resolve_workspace_root",
    "team_api_key_secret_name",
    "validate_secret_references",
    "validate_workspace_root",
]
