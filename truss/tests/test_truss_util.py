import os
import time

from truss import load
from truss.local.local_config_handler import LocalConfigHandler
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


def test_load_trussignore_patterns():
    patterns = path.load_trussignore_patterns(path.TRUSS_IGNORE_PATH)

    assert isinstance(patterns, list)
    assert "__pycache__/" in patterns
    assert ".DS_Store" in patterns
    assert ".git/" in patterns


def test_is_ignored(custom_model_truss_dir_with_hidden_files):
    patterns = path.load_trussignore_patterns(path.TRUSS_IGNORE_PATH)

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


def test_removing_from_gathered_truss_not_original_truss(
    custom_model_truss_dir_with_hidden_files,
):
    # Call a function that is wrapped in a proxy_to_shadow decorator
    # so that we mimic the behavior of a gathered truss
    tr = load(custom_model_truss_dir_with_hidden_files)
    _ = tr.docker_predict({"x": 1})

    # Calculate the shadow path for the original truss
    shadow_truss_dir_name = path.calc_shadow_truss_dirname(
        custom_model_truss_dir_with_hidden_files
    )
    shadow_truss_path = (
        LocalConfigHandler.shadow_trusses_dir_path() / shadow_truss_dir_name
    )

    assert (shadow_truss_path).exists()

    assert not (shadow_truss_path / "__pycache__" / "test.cpython-38.pyc").exists()
    assert not (shadow_truss_path / ".DS_Store").exists()
    assert not (shadow_truss_path / ".git").exists()

    assert (
        custom_model_truss_dir_with_hidden_files / "__pycache__" / "test.cpython-38.pyc"
    ).exists()
    assert (custom_model_truss_dir_with_hidden_files / ".DS_Store").exists()
    assert (custom_model_truss_dir_with_hidden_files / ".git").exists()

    tr.kill_container()


test_max_modified()
