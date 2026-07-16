import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from click.testing import CliRunner

from truss.base.constants import WORKSTATION_TEMPLATE_DIR
from truss.cli.cli import truss_cli
from truss.cli.train.workstation import (
    SUPPORTED_WORKSTATION_ACCELERATORS,
    build_workstation_project,
    copy_workstation_templates,
)
from truss.remote.baseten.custom_types import TeamType
from truss.remote.baseten.remote import BasetenRemote
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


class TestWorkstationTeamResolution:
    @staticmethod
    def _setup_mock_remote(teams, existing_projects=None):
        mock_remote = Mock(spec=BasetenRemote)
        mock_api = Mock()
        mock_remote.api = mock_api
        mock_api.get_teams.return_value = {
            name: TeamType(**team_data) for name, team_data in teams.items()
        }
        mock_api.list_training_projects.return_value = existing_projects or []
        return mock_remote

    @staticmethod
    def _invoke_workstation(runner, team_name=None):
        args = ["train", "workstation", "--remote", "test_remote"]
        if team_name:
            args.extend(["--team", team_name])
        return runner.invoke(truss_cli, args)

    @patch("truss_train.public_api.push")
    @patch("truss.cli.train_commands.RemoteFactory.get_remote_team")
    @patch("truss.cli.train_commands.RemoteFactory.create")
    def test_team_provided_passes_team_id_to_push(
        self, mock_remote_factory, mock_get_remote_team, mock_push
    ):
        teams = {
            "team-a": {"id": "team1", "name": "team-a", "default": True},
            "team-b": {"id": "team2", "name": "team-b", "default": False},
        }
        mock_remote_factory.return_value = self._setup_mock_remote(teams)
        mock_get_remote_team.return_value = None
        mock_push.return_value = {
            "id": "job123",
            "training_project": {"id": "proj123", "name": "workstation-H100"},
        }

        result = self._invoke_workstation(CliRunner(), team_name="team-b")

        assert result.exit_code == 0, result.output
        assert mock_push.call_args[1]["team_id"] == "team2"

    @patch("truss_train.public_api.push")
    @patch("truss.cli.train_commands.RemoteFactory.get_remote_team")
    @patch("truss.cli.train_commands.RemoteFactory.create")
    def test_single_team_auto_resolves_without_flag(
        self, mock_remote_factory, mock_get_remote_team, mock_push
    ):
        teams = {"only-team": {"id": "team9", "name": "only-team", "default": False}}
        mock_remote_factory.return_value = self._setup_mock_remote(teams)
        mock_get_remote_team.return_value = None
        mock_push.return_value = {
            "id": "job123",
            "training_project": {"id": "proj123", "name": "workstation-H100"},
        }

        result = self._invoke_workstation(CliRunner())

        assert result.exit_code == 0, result.output
        assert mock_push.call_args[1]["team_id"] == "team9"

    @patch("truss_train.public_api.push")
    @patch("truss.cli.train_commands.RemoteFactory.get_remote_team")
    @patch("truss.cli.train_commands.RemoteFactory.create")
    def test_invalid_team_errors_before_push(
        self, mock_remote_factory, mock_get_remote_team, mock_push
    ):
        teams = {"team-a": {"id": "team1", "name": "team-a", "default": True}}
        mock_remote_factory.return_value = self._setup_mock_remote(teams)
        mock_get_remote_team.return_value = None

        result = self._invoke_workstation(CliRunner(), team_name="nonexistent")

        assert result.exit_code == 1
        assert "does not exist" in result.output
        mock_push.assert_not_called()
