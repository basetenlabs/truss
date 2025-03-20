import pathlib
import subprocess
import sys
from enum import Enum
from typing import Optional

import pydantic

import truss


class DeployedChainlet(pydantic.BaseModel):
    name: str
    is_entrypoint: bool
    is_draft: bool
    status: str
    logs_url: str
    oracle_name: str


class ChainletArtifact(pydantic.BaseModel):
    truss_dir: pathlib.Path
    display_name: str
    name: str


class ModelOrigin(Enum):
    BASETEN = "BASETEN"
    CHAINS = "CHAINS"


class OracleData(pydantic.BaseModel):
    class Config:
        protected_namespaces = ()

    model_name: str
    s3_key: str
    encoded_config_str: str
    semver_bump: Optional[str] = "MINOR"
    version_name: Optional[str] = None


# This corresponds to `ChainletInputAtomicGraphene` in the backend.
class ChainletDataAtomic(pydantic.BaseModel):
    name: str
    oracle: OracleData


class GitInfo(pydantic.BaseModel):
    latest_commit_sha: str
    latest_tag: Optional[str]
    commits_since_tag: Optional[int]
    has_uncommitted_changes: bool

    @classmethod
    def collect(cls, git_working_dir: pathlib.Path) -> Optional["GitInfo"]:
        def run_git_command(*args):
            try:
                return subprocess.check_output(
                    ["git", *args],
                    text=True,
                    stderr=subprocess.DEVNULL,
                    cwd=git_working_dir,
                ).strip()
            except subprocess.CalledProcessError:
                return None

        latest_commit_sha = run_git_command("rev-parse", "HEAD")
        if not latest_commit_sha:
            return None  # Not inside a git repo

        latest_tag = run_git_command("describe", "--tags", "--abbrev=0") or None
        commits_since_tag = (
            run_git_command("rev-list", f"{latest_tag}..HEAD", "--count")
            if latest_tag
            else None
        )
        has_uncommitted_changes = bool(run_git_command("status", "--porcelain"))

        return cls(
            latest_commit_sha=latest_commit_sha,
            latest_tag=latest_tag,
            commits_since_tag=int(commits_since_tag) if commits_since_tag else None,
            has_uncommitted_changes=has_uncommitted_changes,
        )


class TrussUserEnv(pydantic.BaseModel):
    truss_client_version: str
    python_version: str
    pydantic_version: str
    mypy_version: Optional[str]
    git_info: Optional[GitInfo]

    @classmethod
    def collect(cls) -> "TrussUserEnv":
        py_version = sys.version_info
        try:
            import mypy.version

            mypy_version = mypy.version.__version__
        except ImportError:
            mypy_version = None

        return cls(
            truss_client_version=truss.version(),
            python_version=f"{py_version.major}.{py_version.minor}.{py_version.micro}",
            pydantic_version=pydantic.version.VERSION,
            mypy_version=mypy_version,
            git_info=None,
        )

    @classmethod
    def collect_with_git_info(cls, git_working_dir: pathlib.Path) -> "TrussUserEnv":
        instance = cls.collect()
        instance.git_info = GitInfo.collect(git_working_dir)
        return instance


class BlobType(Enum):
    MODEL = "model"
    TRAIN = "train"
