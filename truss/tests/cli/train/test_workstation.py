import pytest

from truss.cli.train.workstation import build_workstation_project
from truss_train.definitions import (
    InteractiveSessionProvider,
    InteractiveSessionTrigger,
)


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
