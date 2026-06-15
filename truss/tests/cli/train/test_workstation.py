import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from truss.base.constants import WORKSTATION_TEMPLATE_DIR
from truss.cli.train.workstation import (
    SUPPORTED_WORKSTATION_ACCELERATORS,
    build_workstation_project,
    copy_workstation_templates,
)
from truss_train.definitions import (
    InteractiveSessionProvider,
    InteractiveSessionTrigger,
)

requires_posix_bash = pytest.mark.skipif(
    sys.platform == "win32", reason="needs a POSIX bash"
)

EXPECTED_TEMPLATE_FILES = [
    "setup_slurm.sh",
    "install_slurm.sh",
    "setup_controller.sh",
    "setup_worker.sh",
]


def test_build_workstation_project_defaults():
    project = build_workstation_project(
        accelerator="H100", gpu_count=1, project_id="workstation-H100"
    )
    assert project.name == "workstation-H100"

    job = project.job
    assert job.compute.accelerator.accelerator.value == "H100"
    assert job.compute.accelerator.count == 1
    assert job.compute.node_count == 1
    assert job.runtime.start_commands == ["sleep infinity"]
    assert job.interactive_session is not None
    assert job.interactive_session.trigger == InteractiveSessionTrigger.ON_STARTUP
    assert job.interactive_session.session_provider == InteractiveSessionProvider.SSH


@pytest.mark.parametrize("accelerator", SUPPORTED_WORKSTATION_ACCELERATORS)
def test_build_workstation_project_supported_accelerators_multi_gpu(accelerator):
    project = build_workstation_project(
        accelerator=accelerator, gpu_count=4, project_id="my-workstation"
    )
    assert project.name == "my-workstation"

    job = project.job
    assert job.compute.accelerator.accelerator.value == accelerator
    assert job.compute.accelerator.count == 4


def test_build_workstation_project_invalid_accelerator():
    with pytest.raises(ValueError):
        build_workstation_project(accelerator="INVALID", gpu_count=1, project_id="test")


def test_build_workstation_project_multinode_uses_slurm():
    project = build_workstation_project(
        accelerator="H100",
        gpu_count=8,
        project_id="test",
        node_count=4,
        orchestrator="slurm",
    )
    job = project.job
    assert job.compute.node_count == 4
    assert job.runtime.start_commands == ["bash /b10/workspace/setup_slurm.sh"]


def test_copy_workstation_templates():
    with tempfile.TemporaryDirectory() as tmp_dir:
        copy_workstation_templates(Path(tmp_dir))
        for name in EXPECTED_TEMPLATE_FILES:
            script = Path(tmp_dir) / name
            assert script.exists(), f"Missing {name}"
            assert os.access(script, os.X_OK), f"{name} is not executable"


def test_workstation_template_dir_exists():
    assert WORKSTATION_TEMPLATE_DIR.exists()
    for name in EXPECTED_TEMPLATE_FILES:
        assert (WORKSTATION_TEMPLATE_DIR / name).exists(), f"Missing template {name}"


def _eval_slurm_dir(env: dict) -> subprocess.CompletedProcess:
    # Run just the guard + assignment from install_slurm.sh, not the apt/munge install.
    lines = (WORKSTATION_TEMPLATE_DIR / "install_slurm.sh").read_text().splitlines()
    start = next(
        i
        for i, line in enumerate(lines)
        if line.startswith('if [ -z "${BT_TRAINING_JOB_ID}')
    )
    end = next(i for i, line in enumerate(lines) if line.startswith("SLURM_DIR="))
    snippet = "\n".join(lines[start : end + 1])
    return subprocess.run(
        ["bash", "-c", f'{snippet}\necho "$SLURM_DIR"'],
        capture_output=True,
        text=True,
        env=env,
    )


@requires_posix_bash
def test_slurm_rendezvous_dir_is_job_scoped():
    cache = "/root/.cache/user_artifacts"
    job_a = _eval_slurm_dir(
        {"BT_PROJECT_CACHE_DIR": cache, "BT_TRAINING_JOB_ID": "wdgep4w"}
    )
    job_b = _eval_slurm_dir(
        {"BT_PROJECT_CACHE_DIR": cache, "BT_TRAINING_JOB_ID": "3125g1w"}
    )

    assert job_a.stdout.strip() == f"{cache}/slurm_wdgep4w"
    assert job_b.stdout.strip() == f"{cache}/slurm_3125g1w"


@requires_posix_bash
def test_slurm_rendezvous_dir_fails_without_job_id():
    result = _eval_slurm_dir({"BT_PROJECT_CACHE_DIR": "/root/.cache/user_artifacts"})
    assert result.returncode != 0
