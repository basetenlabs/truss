import os

from truss.templates.control.control.helpers.context_managers import current_directory


def test_current_directory(tmp_path):
    orig_cwd = os.getcwd()
    with current_directory(tmp_path):
        assert os.getcwd() == str(tmp_path)

    assert os.getcwd() == orig_cwd
