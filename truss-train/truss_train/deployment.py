import logging
import pathlib
import shutil
import tempfile
import time
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

from truss.base.custom_types import SafeModel
from truss.cli.utils.output import console
from truss.remote.baseten import custom_types as b10_types
from truss.remote.baseten.api import BasetenApi
from truss.remote.baseten.core import archive_dir
from truss.remote.baseten.remote import BasetenRemote
from truss.remote.baseten.utils import transfer
from truss_train.definitions import TrainingJob, TrainingProject, Workspace

# 5GB max archive size
MAX_ARCHIVE_SIZE_BYTES = 5 * 1024 * 1024 * 1024


def _resolve_path(path_str: str, base_dir: Path) -> Path:
    """Resolve a path, handling both absolute and relative paths."""
    path = Path(path_str)
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def _validate_workspace_root(workspace_root: Path, config_path: Path) -> None:
    """Validate that config.py is inside workspace_root."""
    try:
        config_path.resolve().relative_to(workspace_root.resolve())
    except ValueError:
        raise ValueError(
            f"config.py ({config_path}) must be inside workspace_root ({workspace_root})"
        )


def _validate_external_dirs(
    external_dirs: List[Path], workspace_root: Path, exclude_dirs: List[str]
) -> List[Path]:
    """
    Validate external_dirs and filter out any inside workspace_root.

    Returns filtered list of external_dirs that are valid.
    Warns (but continues) if an external_dir is inside workspace_root.
    Raises ValueError for name collisions with workspace_root contents.
    """
    exclude_set = set(exclude_dirs)
    workspace_top_level = {
        p.name for p in workspace_root.iterdir() if p.name not in exclude_set
    }
    seen_names: set[str] = set(workspace_top_level)
    valid_dirs: List[Path] = []

    for ext_dir in external_dirs:
        try:
            ext_dir.resolve().relative_to(workspace_root.resolve())
            console.print(
                f"Warning: external_dir '{ext_dir}' is inside workspace_root. "
                f"It should be outside workspace_root or removed from external_dirs. "
                f"Skipping.",
                style="yellow",
            )
            continue
        except ValueError:
            pass

        if ext_dir.name in seen_names:
            raise ValueError(
                f"Name collision: '{ext_dir.name}' conflicts with an existing "
                f"directory or file. Each external_dir must have a unique name."
            )
        seen_names.add(ext_dir.name)
        valid_dirs.append(ext_dir)

    return valid_dirs


def _gather_training_dir(
    config_path: Path, workspace: Optional[Workspace]
) -> Optional[Path]:
    """
    Gather workspace and external directories into a temporary directory for archiving.

    Returns None if no workspace is specified.
    Otherwise, returns path to the gathered directory.
    """
    if not workspace:
        return None

    config_dir = config_path.absolute().parent
    workspace_root = workspace.workspace_root
    external_dirs = workspace.external_dirs
    exclude_dirs = workspace.exclude_dirs

    if not workspace_root and not external_dirs and not exclude_dirs:
        return None

    if workspace_root:
        resolved_workspace_root = _resolve_path(workspace_root, config_dir)
        _validate_workspace_root(resolved_workspace_root, config_path)
    else:
        resolved_workspace_root = config_dir

    console.print(f"Using workspace root: {resolved_workspace_root}")

    resolved_external_dirs = [_resolve_path(d, config_dir) for d in external_dirs]

    if resolved_external_dirs:
        resolved_external_dirs = _validate_external_dirs(
            resolved_external_dirs, resolved_workspace_root, exclude_dirs
        )

    for ext_dir in resolved_external_dirs:
        if not ext_dir.exists():
            raise ValueError(f"external_dir does not exist: {ext_dir}")
        if not ext_dir.is_dir():
            raise ValueError(f"external_dir is not a directory: {ext_dir}")

    temp_dir = Path(tempfile.mkdtemp(prefix="truss_train_"))
    gathered_dir = temp_dir / "gathered"

    exclude_set = set(exclude_dirs)

    def ignore_excluded(directory: str, contents: List[str]) -> List[str]:
        # Only apply excludes at the top level of workspace_root
        if Path(directory).resolve() == resolved_workspace_root.resolve():
            return [c for c in contents if c in exclude_set]
        return []

    shutil.copytree(
        resolved_workspace_root,
        gathered_dir,
        ignore=ignore_excluded if exclude_dirs else None,
    )

    for ext_dir in resolved_external_dirs:
        dest = gathered_dir / ext_dir.name
        shutil.copytree(ext_dir, dest)

    return gathered_dir


class S3Artifact(SafeModel):
    s3_bucket: str
    s3_key: str


# Performs validation/transformation of fields that we don't want to expose
# to the end user via the TrainingJob SDK.
class PreparedTrainingJob(TrainingJob):
    runtime_artifacts: List[S3Artifact] = []
    truss_user_env: Optional[b10_types.TrussUserEnv] = None

    def model_dump(self, *args, **kwargs):
        data = super().model_dump(*args, **kwargs)

        # NB(nikhil): nest our processed artifacts back under runtime
        data["runtime"]["artifacts"] = data.pop("runtime_artifacts")
        return data


def prepare_push(
    api: BasetenApi,
    config: pathlib.Path,
    training_job: TrainingJob,
    truss_user_env: Optional[b10_types.TrussUserEnv] = None,
):
    source_dir = config.absolute().parent

    gather_start = time.time()
    gathered_dir = _gather_training_dir(
        config_path=config, workspace=training_job.workspace
    )
    dir_to_archive = gathered_dir if gathered_dir else source_dir
    gather_elapsed = time.time() - gather_start
    logger.debug(f"Gather took {gather_elapsed:.2f}s")

    archive_start = time.time()
    archive = archive_dir(dir_to_archive)
    archive_size = Path(archive.name).stat().st_size
    archive_elapsed = time.time() - archive_start
    logger.debug(
        f"Archive took {archive_elapsed:.2f}s ({archive_size / 1024 / 1024:.2f}MB)"
    )

    if archive_size > MAX_ARCHIVE_SIZE_BYTES:
        size_gb = archive_size / (1024 * 1024 * 1024)
        raise ValueError(
            f"Archive size ({size_gb:.2f}GB) exceeds maximum allowed size (5GB). "
            f"Please reduce the size of your workspace by using 'exclude_dirs' to "
            f"exclude large files/directories, or contact Baseten support for assistance."
        )

    credentials = api.get_blob_credentials(b10_types.BlobType.TRAIN)

    upload_start = time.time()
    transfer.multipart_upload_boto3(
        archive.name,
        credentials["s3_bucket"],
        credentials["s3_key"],
        credentials["creds"],
    )
    upload_elapsed = time.time() - upload_start
    logger.debug(f"Upload took {upload_elapsed:.2f}s")
    return PreparedTrainingJob(
        image=training_job.image,
        runtime=training_job.runtime,
        compute=training_job.compute,
        name=training_job.name,
        runtime_artifacts=[
            S3Artifact(s3_key=credentials["s3_key"], s3_bucket=credentials["s3_bucket"])
        ],
        truss_user_env=truss_user_env,
    )


def _upsert_project_and_create_job(
    remote_provider: BasetenRemote,
    training_project: TrainingProject,
    config: Path,
    team_id: Optional[str] = None,
) -> dict:
    project_resp = remote_provider.upsert_training_project(
        training_project=training_project, team_id=team_id
    )

    # Collect TrussUserEnv with git info from the config directory
    working_dir = config.absolute().parent
    truss_user_env = b10_types.TrussUserEnv.collect_with_git_info(working_dir)

    prepared_job = prepare_push(
        remote_provider.api, config, training_project.job, truss_user_env=truss_user_env
    )

    job_resp = remote_provider.api.create_training_job(
        project_id=project_resp["id"], job=prepared_job
    )
    return job_resp


def create_training_job(
    remote_provider: BasetenRemote,
    config: Path,
    training_project: TrainingProject,
    job_name_from_cli: Optional[str] = None,
    team_name: Optional[str] = None,
    team_id: Optional[str] = None,
) -> dict:
    if job_name_from_cli:
        if training_project.job.name:
            console.print(
                f"Warning: name '{training_project.job.name}' provided in config file "
                f"will be ignored. Using job name '{job_name_from_cli}' provided via "
                f"--job-name flag.",
                style="yellow",
            )
        training_project.job.name = job_name_from_cli
    if team_name:
        training_project.team_name = team_name

    job_resp = _upsert_project_and_create_job(
        remote_provider=remote_provider,
        training_project=training_project,
        config=config,
        team_id=team_id,
    )
    job_resp["job_object"] = training_project.job
    return job_resp
