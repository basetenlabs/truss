import os
import time

from truss.util import path


def test_max_modified():
    epoch_time = int(time.time())
    with path.given_or_temporary_dir() as dir:
        time.sleep(0.1)
        t1 = path.get_max_modified_time_of_dir(dir)
        assert t1 > epoch_time
        time.sleep(0.1)
        os.makedirs(os.path.join(dir, "test"))
        t2 = path.get_max_modified_time_of_dir(dir)
        assert t2 > t1


def test_load_gitignore_patterns():
    patterns = path.load_gitignore_patterns(path.TRUSS_GITINORE_PATH)

    assert isinstance(patterns, list)
    assert "__pycache__/" in patterns
    assert ".DS_Store" in patterns
    assert ".git/" in patterns


def test_is_ignored(custom_model_truss_dir_with_hidden_files):
    patterns = path.load_gitignore_patterns(path.TRUSS_GITINORE_PATH)

    assert path.is_ignored(
        custom_model_truss_dir_with_hidden_files
        / "__pycache__"
        / "test.cpython-38.pyc",
        patterns,
    )

    assert path.is_ignored(
        custom_model_truss_dir_with_hidden_files / ".DS_Store", patterns
    )
    assert path.is_ignored(custom_model_truss_dir_with_hidden_files / ".git", patterns)
    assert path.is_ignored(
        custom_model_truss_dir_with_hidden_files / ".git" / ".test_file", patterns
    )
    assert not path.is_ignored(
        custom_model_truss_dir_with_hidden_files / "model", patterns
    )
    assert not path.is_ignored(
        custom_model_truss_dir_with_hidden_files / "model" / "model.py", patterns
    )


def test_remove_ignored_files(custom_model_truss_dir_with_hidden_files):
    path.remove_ignored_files(custom_model_truss_dir_with_hidden_files)
    assert not (
        custom_model_truss_dir_with_hidden_files / "__pycache__" / "test.cpython-38.pyc"
    ).exists()
    assert not (custom_model_truss_dir_with_hidden_files / ".DS_Store").exists()
    assert not (custom_model_truss_dir_with_hidden_files / ".git").exists()
    assert (custom_model_truss_dir_with_hidden_files / "model").exists()
    assert (custom_model_truss_dir_with_hidden_files / "model" / "model.py").exists()


test_max_modified()
test_load_gitignore_patterns
test_is_ignored()
test_remove_ignored_files()
