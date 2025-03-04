import pathlib

import pytest

from truss_train import loader

TEST_ROOT = pathlib.Path(__file__).parent.resolve()


def test_import_requires_training_project():
    job_src = TEST_ROOT / "import" / "config_without_training_project.py"
    match = r"No `.+` was found."
    with pytest.raises(ValueError, match=match):
        with loader.import_target(job_src):
            pass


def test_import_requires_single_training_project():
    job_src = TEST_ROOT / "import" / "config_with_multiple_training_projects.py"
    match = r"Multiple `.+`s were found."
    with pytest.raises(ValueError, match=match):
        with loader.import_target(job_src):
            pass


def test_import_with_single_training_project():
    job_src = TEST_ROOT / "import" / "config_with_single_training_project.py"
    with loader.import_target(job_src) as training_project:
        assert training_project.name == "first-project"
        assert training_project.job.compute.cpu_count == 4


def test_import_directory_fails():
    job_src = TEST_ROOT / "import"
    match = r"You must point to a python file"
    with pytest.raises(ImportError, match=match):
        with loader.import_target(job_src):
            pass
