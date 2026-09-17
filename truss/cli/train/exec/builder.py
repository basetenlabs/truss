"""Assembles a `TrainingProject` for a directory-and-command exec job."""

import shlex
from pathlib import Path
from typing import List, Mapping, Optional, Sequence, Union

import rich_click as click

from truss.base import truss_config
from truss.cli.train import workstation
from truss_train.definitions import (
    CacheConfig,
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

from .project import Project

# A CPU-only job doesn't need a CUDA image.
PYTHON_BASE_IMAGE = "python:3.12-slim"

# An empty project name fails server-side validation, which the filesystem root
# would otherwise produce.
DEFAULT_EXEC_PROJECT_NAME = "truss-train-exec"

SUPPORTED_EXEC_ACCELERATORS = workstation.SUPPORTED_WORKSTATION_ACCELERATORS

# Read from the model so a caller that wants the platform default has one. Named
# for exec because `truss_config` exports a `DEFAULT_MEMORY` with a different value.
DEFAULT_EXEC_CPU_COUNT: int = Compute.model_fields["cpu_count"].default
DEFAULT_EXEC_MEMORY: str = Compute.model_fields["memory"].default


def default_base_image(accelerator: Optional[str], project: Optional[Project]) -> str:
    """The base image to use when the user didn't pass --image.

    An accelerator wins over the project's preference: a GPU job needs the CUDA image
    regardless of how the project installs its dependencies, and the project's setup
    steps cover the difference.
    """
    if accelerator is not None:
        return workstation.default_base_image(accelerator)
    if project is not None:
        return project.base_image()
    return PYTHON_BASE_IMAGE


def resolve_workspace_root(source_dir: Path, workspace_root: Optional[str]) -> Path:
    """The directory that actually gets archived and becomes the job's root."""
    if not workspace_root:
        return source_dir
    root = Path(workspace_root)
    if not root.is_absolute():
        root = source_dir / root
    return root.resolve()


def validate_workspace_root(source_dir: Path, workspace_root: Optional[str]) -> Path:
    """Resolve and check `--workspace-root`, returning the effective job root.

    `truss_train` runs the same containment check inside `push`, but only after the
    training project has been created, so a bad value there leaves a stray empty
    project behind. Checking here keeps that from happening.
    """
    root = resolve_workspace_root(source_dir, workspace_root)
    if not workspace_root:
        return root

    if not root.is_dir():
        raise click.UsageError(
            f"--workspace-root '{workspace_root}' resolves to {root}, "
            "which is not a directory."
        )
    try:
        source_dir.resolve().relative_to(root)
    except ValueError:
        raise click.UsageError(
            f"--workspace-root '{workspace_root}' resolves to {root}, which does not "
            f"contain the current directory ({source_dir}); it must be a parent of it."
        )
    return root


# The working directory and the uv cache have to share a filesystem. uv hardlinks
# wheels out of its cache into the venv and silently falls back to full copies when
# the two are on different mounts, which stores the whole dependency set twice --
# enough on its own to exceed a job's ephemeral-storage limit. uv's default cache
# (`$HOME/.cache/uv`) lands under `CacheConfig.mount_base_path`, which is the
# network-backed cache volume when one is mounted, so it always differs from the
# working directory unless it is moved.
#
# It stays on the working directory rather than the project cache volume, which is
# network-backed: a package cache is latency-bound, and the volume is the right place
# for datasets and weights instead.
UV_CACHE_SUBDIR = ".uv-cache"

_LOCAL_UV_CACHE_STEP = f'export UV_CACHE_DIR="$PWD/{UV_CACHE_SUBDIR}"'


def build_start_commands(
    start_command: Sequence[str], setup_steps: Sequence[str] = ()
) -> List[str]:
    """Build `Runtime.start_commands` for `start_command`.

    The user's command always runs last and verbatim. Steps that prepare the
    environment run first -- co-locating the uv cache with the working directory,
    then `setup_steps` from the detected project type -- chained into a single
    `/bin/sh -c` entry to match the pattern in `truss/templates/train/config.py`, so
    that an `export` is still in effect when the command runs.
    """
    command = shlex.join(start_command)
    steps = [_LOCAL_UV_CACHE_STEP, *setup_steps]
    return [f"/bin/sh -c {shlex.quote(' && '.join([*steps, command]))}"]


def build_exec_project(
    *,
    start_command: Sequence[str],
    project_name: str,
    accelerator: Optional[str],
    gpu_count: int,
    cpu_count: int,
    memory: str,
    base_image: Optional[str],
    project: Optional[Project],
    workspace_root: Optional[str],
    exclude_dirs: Sequence[str],
    external_dirs: Sequence[str],
    environment_variables: Mapping[str, Union[str, SecretReference]],
    enable_cache: bool,
) -> TrainingProject:
    """Build the training project for an exec job.

    Every parameter is required and keyword-only, so a caller cannot build a
    partially-specified project and the CLI stays the single source of defaults. The
    `Optional` types are values, not omissions: `accelerator` None means CPU-only,
    `base_image` None means derive it, `project` None means run against a plain image
    with no setup steps, and `workspace_root` None means archive the invocation
    directory.
    """
    accelerator_spec = None
    if accelerator is not None:
        accelerator_spec = truss_config.AcceleratorSpec(
            accelerator=truss_config.Accelerator(accelerator), count=gpu_count
        )

    compute = Compute(cpu_count=cpu_count, memory=memory, accelerator=accelerator_spec)

    resolved_base_image = base_image or default_base_image(accelerator, project)

    # Checkpointing stays off: a one-off command has no checkpoints to write. The
    # cache volume is the caller's call -- it is where data a rerun should not
    # re-download lives, which only some workloads have.
    runtime = Runtime(
        start_commands=build_start_commands(
            start_command=start_command,
            setup_steps=project.setup(resolved_base_image) if project else (),
        ),
        environment_variables=dict(environment_variables),
        cache_config=CacheConfig(enabled=True, require_cache_affinity=False)
        if enable_cache
        else None,
    )

    # SSH available on demand, rather than a session live from job startup: the
    # session timeout applies once the job ends, so it is not a concern for a
    # long-running job. No timeout is set here; the model default still applies.
    interactive_session = InteractiveSession(
        trigger=InteractiveSessionTrigger.ON_DEMAND,
        session_provider=InteractiveSessionProvider.SSH,
    )

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
