import pathlib

import pytest

from truss_train import loader
from truss_train.definitions import TrainingProject, DeployCheckpointsConfig

TEST_ROOT = pathlib.Path(__file__).parent.resolve()


def test_import_requires_training_project():
    job_src = TEST_ROOT / "import" / "config_without_training_project.py"
    match = r"No `.+` was found."
    with pytest.raises(ValueError, match=match):
        with loader.import_target(job_src, TrainingProject):
            pass


def test_import_requires_single_training_project():
    job_src = TEST_ROOT / "import" / "config_with_multiple_training_projects.py"
    match = r"Multiple `.+`s were found."
    with pytest.raises(ValueError, match=match):
        with loader.import_target(job_src, TrainingProject):
            pass


def test_import_with_single_training_project():
    job_src = TEST_ROOT / "import" / "config_with_single_training_project.py"
    with loader.import_target(job_src, TrainingProject) as training_project:
        assert training_project.name == "first-project"
        assert training_project.job.compute.cpu_count == 4


def test_import_directory_fails():
    job_src = TEST_ROOT / "import"
    match = r"You must point to a python file"
    with pytest.raises(ImportError, match=match):
        with loader.import_target(job_src, TrainingProject):
            pass

def test_import_deploy_checkpoints_config():
    job_src = TEST_ROOT / "import" / "deploy_checkpoints_config.py"
    with loader.import_target(job_src, DeployCheckpointsConfig) as deploy_checkpoints_config:
        assert len(deploy_checkpoints_config.checkpoint_details.checkpoints) == 2
        assert deploy_checkpoints_config.checkpoint_details.base_model_id == "unsloth/gemma-3-1b-it"
        assert deploy_checkpoints_config.checkpoint_details.checkpoints[0].id == "checkpoint-24"
        assert deploy_checkpoints_config.checkpoint_details.checkpoints[1].id == "checkpoint-42"
    
def test_import_handles_training_project_with_deploy_checkpoints_config():
    job_src = TEST_ROOT / "import" / "project_with_deploy_checkpoints_config.py"
    with loader.import_target(job_src, TrainingProject) as training_project:
        assert training_project.name == "first-project"
        assert training_project.job.compute.cpu_count == 4
    with loader.import_target(job_src, DeployCheckpointsConfig) as deploy_checkpoints_config:
        assert len(deploy_checkpoints_config.checkpoint_details.checkpoints) == 2
        assert deploy_checkpoints_config.checkpoint_details.base_model_id == "unsloth/gemma-3-1b-it"
        assert deploy_checkpoints_config.checkpoint_details.checkpoints[0].id == "checkpoint-24"
        assert deploy_checkpoints_config.checkpoint_details.checkpoints[1].id == "checkpoint-42"
