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
    patterns = path.load_trussignore_patterns(path.FIXED_TRUSS_IGNORE_PATH)

    assert isinstance(patterns, list)
    assert "__pycache__/" in patterns
    assert ".DS_Store" in patterns
    assert ".git/" in patterns


def test_are_dirs_equal(custom_model_truss_dir):
    assert path.are_dirs_equal(custom_model_truss_dir, custom_model_truss_dir)
    assert not path.are_dirs_equal(
        custom_model_truss_dir, custom_model_truss_dir / "model"
    )

    with path.given_or_temporary_dir() as dir:
        path.copy_tree_path(custom_model_truss_dir, dir)
        assert path.are_dirs_equal(custom_model_truss_dir, dir)
        (custom_model_truss_dir / "model" / "model.py").write_text("print('hello')")
        assert not path.are_dirs_equal(custom_model_truss_dir, dir)
        # Test the modification time logic
        (custom_model_truss_dir / "model" / "model.py").write_text("print('hello')")
        assert not path.are_dirs_equal(custom_model_truss_dir, dir)


def test_copy_tree_path_with_no_skipping(custom_model_truss_dir):
    with path.given_or_temporary_dir() as dir:
        path.copy_tree_path(custom_model_truss_dir, dir)

        for file in custom_model_truss_dir.rglob("*"):
            assert (dir / file.relative_to(custom_model_truss_dir)).exists()


def test_copy_tree_path_with_skipping(custom_model_truss_dir):
    with path.given_or_temporary_dir() as dir:
        path.copy_tree_path(custom_model_truss_dir, dir, directories_to_skip=["model"])

        for file in custom_model_truss_dir.rglob("*"):
            relative_file_path = file.relative_to(custom_model_truss_dir)
            if relative_file_path.parts[0] == "model":
                assert not (dir / relative_file_path).exists()
            else:
                assert (dir / relative_file_path).exists()


def test_remove_tree_path_with_no_skipping(custom_model_truss_dir):
    path.remove_tree_path(custom_model_truss_dir)
    assert not custom_model_truss_dir.exists()


def test_remove_tree_path_with_skipping(custom_model_truss_dir):
    path.remove_tree_path(custom_model_truss_dir, directories_to_skip=["model"])
    assert custom_model_truss_dir.exists()
    assert (custom_model_truss_dir / "model").exists()
    assert (custom_model_truss_dir / "model" / "model.py").exists()


def test_is_ignored(custom_model_truss_dir_with_hidden_files):
    patterns = path.load_trussignore_patterns(path.FIXED_TRUSS_IGNORE_PATH)

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


def test_is_ignored_with_base_dir(custom_model_truss_dir_with_hidden_files):
    patterns = path.load_trussignore_patterns(path.FIXED_TRUSS_IGNORE_PATH)

    # Grab a directory that is a parent of the truss directory to test the base_dir functionality
    random_parent_directory = custom_model_truss_dir_with_hidden_files.parent.name

    # By adding the random_parent_director to the patterns, without base_dir, every file should be
    # ignored in the truss directory because a part of the path is now in the patterns.
    # However, with the base_dir, the random_parent_directory shouldn't be present in the path
    # that is_ignored uses to check if the file is ignored.
    patterns.append(random_parent_directory)

    assert path.is_ignored(
        custom_model_truss_dir_with_hidden_files / ".git" / ".test_file", patterns
    )

    assert path.is_ignored(
        custom_model_truss_dir_with_hidden_files / ".git" / ".test_file",
        patterns,
        base_dir=custom_model_truss_dir_with_hidden_files,
    )

    assert path.is_ignored(custom_model_truss_dir_with_hidden_files / "model", patterns)

    assert not path.is_ignored(
        custom_model_truss_dir_with_hidden_files / "model",
        patterns,
        base_dir=custom_model_truss_dir_with_hidden_files,
    )

    assert path.is_ignored(
        custom_model_truss_dir_with_hidden_files / "model" / "model.py", patterns
    )

    assert not path.is_ignored(
        custom_model_truss_dir_with_hidden_files / "model" / "model.py",
        patterns,
        base_dir=custom_model_truss_dir_with_hidden_files,
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
    _ = tr.kill_container()

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
    assert (shadow_truss_path / "model").exists()

    assert (
        custom_model_truss_dir_with_hidden_files / "__pycache__" / "test.cpython-38.pyc"
    ).exists()
    assert (custom_model_truss_dir_with_hidden_files / ".DS_Store").exists()
    assert (custom_model_truss_dir_with_hidden_files / ".git").exists()
    assert (custom_model_truss_dir_with_hidden_files / "model").exists()


def test_skipping_data_directory_when_files_are_equal(
    custom_model_truss_dir_with_hidden_files,
):
    """
    This test tests the case where the data directory is skipped because the
    data directory in the original truss is the same as the data directory in
    the shadow truss.

    We then test that the data directory is not skipped when the data directory
    in the original truss is different from the data directory in the shadow
    truss.
    """
    tr = load(custom_model_truss_dir_with_hidden_files)

    # Call a function that is wrapped in a proxy_to_shadow decorator
    # so that we mimic the behavior of a gathered truss
    _ = tr.kill_container()

    shadow_truss_dir_name = path.calc_shadow_truss_dirname(
        custom_model_truss_dir_with_hidden_files
    )
    shadow_truss_path = (
        LocalConfigHandler.shadow_trusses_dir_path() / shadow_truss_dir_name
    )

    shadow_data_file = shadow_truss_path / "data" / "test_file"
    original_data_file = custom_model_truss_dir_with_hidden_files / "data" / "test_file"
    shadow_data_file_modified_time = shadow_data_file.stat().st_mtime

    assert (shadow_truss_path / "data").exists()
    assert (shadow_truss_path / "data" / "test_file").exists()

    # Create a random file to
    # 1. Update the max modified time so that the gather process actually gathers the data directory
    # 2. Test that the copy operation did occur on the second gather
    (custom_model_truss_dir_with_hidden_files / "model" / "sample_file.py").touch()

    _ = tr.kill_container()

    shadow_data_file_modified_time_after_second_gather = (
        shadow_data_file.stat().st_mtime
    )

    # During the second gather op, the data directory should not have been copied because the data directory
    # in the original truss never changed from the first gather op
    assert (
        shadow_data_file_modified_time_after_second_gather
        == shadow_data_file_modified_time
    )
    assert (shadow_truss_path / "model" / "sample_file.py").exists()

    # Update the original data file
    original_data_file.write_text("test")

    _ = tr.kill_container()

    shadow_data_file_modified_time_after_third_gather = shadow_data_file.stat().st_mtime

    assert (
        shadow_data_file_modified_time_after_third_gather
        != shadow_data_file_modified_time
    )

    assert (
        shadow_data_file_modified_time_after_third_gather
        != shadow_data_file_modified_time_after_second_gather
    )

    assert shadow_data_file.read_text() == "test"


test_max_modified()
