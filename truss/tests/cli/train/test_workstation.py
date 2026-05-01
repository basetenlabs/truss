import os
import tempfile
from pathlib import Path

import pytest

from truss.cli.train.workstation import (
    build_workstation_project,
    copy_workstation_templates,
)
from truss_train.definitions import (
    InteractiveSessionProvider,
    InteractiveSessionTrigger,
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


def test_build_workstation_project_h200_multi_gpu():
    project = build_workstation_project(
        accelerator="H200", gpu_count=4, project_id="my-workstation"
    )
    assert project.name == "my-workstation"

    job = project.job
    assert job.compute.accelerator.accelerator.value == "H200"
    assert job.compute.accelerator.count == 4


def test_build_workstation_project_invalid_accelerator():
    with pytest.raises(ValueError):
        build_workstation_project(accelerator="INVALID", gpu_count=1, project_id="test")


def test_build_workstation_project_multinode_uses_slurm():
    project = build_workstation_project(
        accelerator="H100", gpu_count=8, project_id="test", node_count=4
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
    from truss.base.constants import WORKSTATION_TEMPLATE_DIR

    assert WORKSTATION_TEMPLATE_DIR.exists()
    for name in EXPECTED_TEMPLATE_FILES:
        assert (WORKSTATION_TEMPLATE_DIR / name).exists(), f"Missing template {name}"
