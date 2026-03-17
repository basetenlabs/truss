"""Core logic for SLURM harness CLI commands."""

import json
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from truss.base.constants import SLURM_HARNESS_TEMPLATE_DIR

DEFAULT_PARTITION = "H200"
DEFAULT_BASE_IMAGE = "pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime"

# Path to runtime_config.json on a login node (written by push before push)
WORKSPACE_RUNTIME_CONFIG = Path("/workspace/runtime_config.json")


def _read_login_config() -> dict:
    """Read runtime_config.json from the login node workspace, or empty dict."""
    try:
        return json.loads(WORKSPACE_RUNTIME_CONFIG.read_text())
    except (json.JSONDecodeError, KeyError, OSError):
        return {}


def detect_default_project() -> str:
    """Read project name from /workspace/runtime_config.json if on a login node."""
    return _read_login_config().get("project_name", "slurm-harness")


def detect_login_image() -> Optional[str]:
    """Read base_image from the login node's runtime_config, or None."""
    return _read_login_config().get("base_image")


def parse_gres(gres_str: str) -> int:
    """Parse --gres=gpu:N format and return GPU count."""
    if not gres_str:
        return 8
    parts = gres_str.split(":")
    if len(parts) == 2 and parts[0] == "gpu":
        return int(parts[1])
    elif len(parts) == 1:
        return int(parts[0])
    else:
        return 8


def build_login_runtime_config(
    project: str,
    gpus_per_node: int,
    partition: Optional[str],
    self_test: bool,
    image: Optional[str] = None,
) -> dict:
    """Build runtime_config dict for the login node."""
    config: dict = {
        "project_name": project,
        "job_name": "slurm-login",
        "gpus_per_node": gpus_per_node,
        "self_test": self_test,
    }
    if partition:
        config["partition"] = partition
    if image:
        config["base_image"] = image
    return config


def build_sbatch_runtime_config(
    project: str,
    job_name: str,
    node_count: int,
    gpus_per_node: int,
    partition: str,
    sbatch_script: str,
    image: Optional[str] = None,
) -> dict:
    """Build runtime_config dict for the worker node."""
    config: dict = {
        "project_name": project,
        "job_name": job_name,
        "node_count": node_count,
        "gpus_per_node": gpus_per_node,
        "partition": partition,
        "sbatch_script": sbatch_script,
    }
    if image:
        config["base_image"] = image
    return config


def push_node(node_type: str, runtime_config: dict, remote: str = "baseten") -> dict:
    """
    Push a SLURM harness node to Baseten training.

    Args:
        node_type: One of "login_node", "worker_node", "seed"
        runtime_config: Dict written to runtime_config.json
        remote: Baseten remote target
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        shutil.copytree(
            SLURM_HARNESS_TEMPLATE_DIR,
            tmp_path,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git", ".gitignore"),
        )

        (tmp_path / "runtime_config.json").write_text(
            json.dumps(runtime_config, indent=2)
        )
        shutil.copy(tmp_path / node_type / "config.py", tmp_path / "config.py")

        from truss_train.public_api import push

        return push(config=tmp_path / "config.py", remote=remote)
