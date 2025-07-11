import pathlib

import pydantic
import pytest

from truss_train import loader

TEST_ROOT = pathlib.Path(__file__).parent.resolve()


def test_import_requires_training_project():
    job_src = TEST_ROOT / "import" / "config_without_training_project.py"
    match = r"No `.+` was found."
    with pytest.raises(ValueError, match=match):
        with loader.import_training_project(job_src):
            pass


def test_import_requires_single_training_project():
    job_src = TEST_ROOT / "import" / "config_with_multiple_training_projects.py"
    match = r"Multiple `.+`s were found."
    with pytest.raises(ValueError, match=match):
        with loader.import_training_project(job_src):
            pass


def test_import_with_single_training_project():
    job_src = TEST_ROOT / "import" / "config_with_single_training_project.py"
    with loader.import_training_project(job_src) as training_project:
        assert training_project.name == "first-project"
        assert training_project.job.compute.cpu_count == 4
        assert training_project.job.runtime.cache_config.enabled
        assert not training_project.job.runtime.cache_config.enable_legacy_hf_mount


def test_import_rejects_extra_fields():
    """Test that importing a config with extra fields raises a validation error."""
    job_src = TEST_ROOT / "import" / "config_with_extra_field.py"
    with pytest.raises(
        pydantic.ValidationError, match="Extra inputs are not permitted"
    ):
        with loader.import_training_project(job_src):
            pass


def test_import_directory_fails():
    job_src = TEST_ROOT / "import"
    match = r"You must point to a python file"
    with pytest.raises(ImportError, match=match):
        with loader.import_training_project(job_src):
            pass


def test_import_deploy_checkpoints_config():
    job_src = TEST_ROOT / "import" / "deploy_checkpoints_config.py"
    with loader.import_deploy_checkpoints_config(job_src) as deploy_checkpoints_config:
        assert len(deploy_checkpoints_config.checkpoint_details.checkpoints) == 2
        assert (
            deploy_checkpoints_config.checkpoint_details.base_model_id
            == "unsloth/gemma-3-1b-it"
        )
        assert (
            deploy_checkpoints_config.checkpoint_details.checkpoints[0].id
            == "checkpoint-24"
        )
        assert (
            deploy_checkpoints_config.checkpoint_details.checkpoints[1].id
            == "checkpoint-42"
        )


def test_import_handles_training_project_with_deploy_checkpoints_config():
    job_src = TEST_ROOT / "import" / "project_with_deploy_checkpoints_config.py"
    with loader.import_training_project(job_src) as training_project:
        assert training_project.name == "first-project"
        assert training_project.job.compute.cpu_count == 4
    with loader.import_deploy_checkpoints_config(job_src) as deploy_checkpoints_config:
        assert len(deploy_checkpoints_config.checkpoint_details.checkpoints) == 2
        assert (
            deploy_checkpoints_config.checkpoint_details.base_model_id
            == "unsloth/gemma-3-1b-it"
        )
        assert (
            deploy_checkpoints_config.checkpoint_details.checkpoints[0].id
            == "checkpoint-24"
        )
        assert (
            deploy_checkpoints_config.checkpoint_details.checkpoints[1].id
            == "checkpoint-42"
        )
