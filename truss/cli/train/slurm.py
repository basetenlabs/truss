"""Core logic for SLURM harness CLI commands."""

import json
import shutil
import tempfile
from pathlib import Path

from truss.base.constants import SLURM_HARNESS_TEMPLATE_DIR

# Canonical accelerator partition names
PARTITIONS = ("H100", "H200", "A100")
DEFAULT_PARTITION = "H200"

# Path to runtime_config.json on a login node (written by push before push)
WORKSPACE_RUNTIME_CONFIG = Path("/workspace/runtime_config.json")


def detect_default_project() -> str:
    """Read project name from /workspace/runtime_config.json if on a login node."""
    if WORKSPACE_RUNTIME_CONFIG.exists():
        try:
            return json.loads(WORKSPACE_RUNTIME_CONFIG.read_text()).get(
                "project_name", "slurm-harness"
            )
        except (json.JSONDecodeError, KeyError):
            pass
    return "slurm-harness"


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
    project: str, workers: int, gpus_per_node: int, partition: str, self_test: bool
) -> dict:
    """Build runtime_config dict for the login node."""
    return {
        "project_name": project,
        "job_name": "slurm-login",
        "node_count": workers,
        "gpus_per_node": gpus_per_node,
        "partition": partition,
        "self_test": self_test,
    }


def build_sbatch_runtime_config(
    project: str,
    job_name: str,
    node_count: int,
    gpus_per_node: int,
    partition: str,
    sbatch_script: str,
) -> dict:
    """Build runtime_config dict for the worker node."""
    return {
        "project_name": project,
        "job_name": job_name,
        "node_count": node_count,
        "gpus_per_node": gpus_per_node,
        "partition": partition,
        "sbatch_script": sbatch_script,
    }


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
