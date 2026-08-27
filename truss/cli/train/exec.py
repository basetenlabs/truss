import shlex
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import rich_click as click

from truss.base import truss_config
from truss.cli.train import workstation
from truss_train.definitions import (
    Compute,
    Image,
    InteractiveSession,
    InteractiveSessionProvider,
    InteractiveSessionTrigger,
    Runtime,
    SecretReference,
    TrainingJob,
    TrainingProject,
    Workspace,
)

# A CPU-only job doesn't need a CUDA image, so default to a slim Python. With
# --with-uv we use the official uv image instead, which is the same slim Python with
# uv preinstalled. (uv stopped publishing bookworm-slim variants for current
# versions, hence trixie-slim.)
PYTHON_BASE_IMAGE = "python:3.12-slim"
UV_BASE_IMAGE = "ghcr.io/astral-sh/uv:0.12.6-python3.12-trixie-slim"

# Fallback project name for the rare case that the current directory has no name
# (e.g. the filesystem root).
DEFAULT_EXEC_PROJECT_NAME = "truss-train-exec"

# The same set workstation supports: every accelerator the platform exposes.
SUPPORTED_EXEC_ACCELERATORS = workstation.SUPPORTED_WORKSTATION_ACCELERATORS

# Mirror the Compute model defaults so `--cpu-count` / `--memory` stay in sync with it.
DEFAULT_CPU_COUNT: int = Compute.model_fields["cpu_count"].default
DEFAULT_MEMORY: str = Compute.model_fields["memory"].default

UV_LOCK_FILE = "uv.lock"
PYPROJECT_FILE = "pyproject.toml"

# Used with --with-uv when the base image may not already ship uv. Skips the install
# when uv is present, so it is safe on an image that has it. The standalone installer
# needs curl, which the CUDA base image has; a custom --image may not.
UV_INSTALL_STEPS = [
    "{ command -v uv >/dev/null 2>&1 || "
    "curl -LsSf https://astral.sh/uv/install.sh | sh ; }",
    'export PATH="$HOME/.local/bin:$PATH"',
]


def default_base_image(accelerator: Optional[str], with_uv: bool = False) -> str:
    if accelerator is not None:
        return workstation.default_base_image(accelerator)
    return UV_BASE_IMAGE if with_uv else PYTHON_BASE_IMAGE


def _parse_key_value_flag(flag: str, expected: str, entry: str) -> Tuple[str, str]:
    # Split on the first `=` only, so values that contain `=` survive intact.
    key, separator, value = entry.partition("=")
    if not separator or not key:
        raise click.UsageError(f"Invalid {flag} value '{entry}'. Expected {expected}.")
    return key, value


def parse_environment_variables(
    env: Sequence[str] = (), secrets: Sequence[str] = ()
) -> Dict[str, Union[str, SecretReference]]:
    """Turn `--env KEY=VALUE` and `--secret KEY=SECRET_NAME` flags into the
    `Runtime.environment_variables` mapping."""
    entries: List[Tuple[str, Union[str, SecretReference]]] = []
    for entry in env:
        key, value = _parse_key_value_flag("--env", "KEY=VALUE", entry)
        entries.append((key, value))
    for entry in secrets:
        key, secret_name = _parse_key_value_flag("--secret", "KEY=SECRET_NAME", entry)
        entries.append((key, SecretReference(name=secret_name)))

    environment_variables: Dict[str, Union[str, SecretReference]] = {}
    for key, value in entries:
        if key in environment_variables:
            raise click.UsageError(
                f"Environment variable '{key}' is set more than once by "
                "--env / --secret."
            )
        environment_variables[key] = value
    return environment_variables


def looks_like_uv_project(source_dir: Path) -> bool:
    """Whether `source_dir` carries uv project metadata, i.e. its command probably
    needs uv present in the job image."""
    return (source_dir / UV_LOCK_FILE).exists() or (
        source_dir / PYPROJECT_FILE
    ).exists()


def build_start_commands(
    start_command: Sequence[str], install_uv: bool = False
) -> List[str]:
    """Build `Runtime.start_commands` for `start_command`.

    The user's command always runs last and verbatim. `install_uv` prepends the uv
    install for the images that may not ship it; the two are chained into a single
    `/bin/sh -c` entry, matching the pattern in `truss/templates/train/config.py`,
    because whether the platform runs more than the first entry can't be established
    from this repo.
    """
    command = shlex.join(start_command)
    if not install_uv:
        return [command]
    return [f"/bin/sh -c {shlex.quote(' && '.join([*UV_INSTALL_STEPS, command]))}"]


def build_exec_project(
    start_command: Sequence[str],
    project_name: str,
    accelerator: Optional[str] = None,
    gpu_count: int = 1,
    cpu_count: int = DEFAULT_CPU_COUNT,
    memory: str = DEFAULT_MEMORY,
    base_image: Optional[str] = None,
    with_uv: bool = False,
    workspace_root: Optional[str] = None,
    exclude_dirs: Sequence[str] = (),
    external_dirs: Sequence[str] = (),
    environment_variables: Optional[Dict[str, Union[str, SecretReference]]] = None,
) -> TrainingProject:
    """Build the training project for `truss train exec`.

    `start_command` is the argv of the command to run in the job. `accelerator`
    being None means a CPU-only job, which is the default.
    """
    accelerator_spec = None
    if accelerator is not None:
        accelerator_spec = truss_config.AcceleratorSpec(
            accelerator=truss_config.Accelerator(accelerator), count=gpu_count
        )

    compute = Compute(cpu_count=cpu_count, memory=memory, accelerator=accelerator_spec)

    resolved_base_image = base_image or default_base_image(accelerator, with_uv)

    # No cache_config, checkpointing_config or load_checkpoint_config: this command
    # runs a one-off command, so it gets no persistent storage and no checkpointing.
    runtime = Runtime(
        start_commands=build_start_commands(
            start_command=start_command,
            # Only the uv base image is known to ship uv; with --with-uv the CUDA
            # image used for GPU jobs and any custom --image get the
            # (skip-if-present) install prepended.
            install_uv=with_uv and resolved_base_image != UV_BASE_IMAGE,
        ),
        environment_variables=dict(environment_variables or {}),
    )

    # SSH available on demand, rather than a session live from job startup: the
    # session timeout applies once the job ends, so it is not a concern for a
    # long-running job. No timeout is set here; the model default still applies.
    interactive_session = InteractiveSession(
        trigger=InteractiveSessionTrigger.ON_DEMAND,
        session_provider=InteractiveSessionProvider.SSH,
    )

    # Left as None unless a directory flag was given, so the plain case just
    # archives the directory the command was invoked from.
    workspace_config = None
    if workspace_root or exclude_dirs or external_dirs:
        workspace_config = Workspace(
            workspace_root=workspace_root,
            exclude_dirs=list(exclude_dirs),
            external_dirs=list(external_dirs),
        )

    job = TrainingJob(
        image=Image(base_image=resolved_base_image),
        compute=compute,
        runtime=runtime,
        interactive_session=interactive_session,
        workspace=workspace_config,
    )

    return TrainingProject(name=project_name, job=job)
