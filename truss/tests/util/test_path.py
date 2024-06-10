import os
import tempfile
import time
from pathlib import Path

from truss import load
from truss.contexts.image_builder.serving_image_builder import (
    ServingImageBuilderContext,
)
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
    assert ".git" in patterns


def test_copy_tree_path_with_no_hidden_files(custom_model_truss_dir):
    with path.given_or_temporary_dir() as dir:
        path.copy_tree_path(custom_model_truss_dir, dir)

        for file in custom_model_truss_dir.rglob("*"):
            assert (dir / file.relative_to(custom_model_truss_dir)).exists()


def test_copy_tree_path_with_hidden_files(custom_model_truss_dir_with_hidden_files):
    with path.given_or_temporary_dir() as dir:
        path.copy_tree_path(custom_model_truss_dir_with_hidden_files, dir)

        assert not (dir / "__pycache__" / "test.cpython-311.pyc").exists()
        assert not (dir / ".DS_Store").exists()
        assert not (dir / ".git").exists()
        assert (dir / "model").exists()
        assert (dir / "model" / "model.py").exists()


def test_is_ignored(custom_model_truss_dir_with_hidden_files):
    patterns = path.load_trussignore_patterns(path.FIXED_TRUSS_IGNORE_PATH)

    assert path.is_ignored(
        custom_model_truss_dir_with_hidden_files
        / "__pycache__"
        / "test.cpython-311.pyc",
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


def test_ignored_files_in_docker_context(
    custom_model_truss_dir_with_hidden_files,
):
    tr = load(custom_model_truss_dir_with_hidden_files)

    with path.given_or_temporary_dir() as dir:
        image_builder = ServingImageBuilderContext.run(tr._truss_dir)
        image_builder.prepare_image_build_dir(dir)

        assert dir.exists()

        assert not (dir / "__pycache__" / "test.cpython-311.pyc").exists()
        assert not (dir / ".DS_Store").exists()
        assert not (dir / ".git").exists()
        assert (dir / "model").exists()

        assert (
            custom_model_truss_dir_with_hidden_files
            / "__pycache__"
            / "test.cpython-311.pyc"
        ).exists()
        assert (custom_model_truss_dir_with_hidden_files / ".DS_Store").exists()
        assert (custom_model_truss_dir_with_hidden_files / ".git").exists()
        assert (custom_model_truss_dir_with_hidden_files / "model").exists()


def test_copy_tree_path_with_truss_ignore(custom_model_truss_dir_with_truss_ignore):
    with tempfile.TemporaryDirectory() as temp_dir:
        dest_dir = Path(temp_dir)
        ignored_files_folders = ["random_folder_1", "random_file_1.txt"]
        path.copy_tree_path(
            custom_model_truss_dir_with_truss_ignore,
            dest_dir,
            ignore_patterns=ignored_files_folders,
        )

        for ignored in ignored_files_folders:
            assert not (dest_dir / ignored).exists()

        assert (dest_dir / "random_folder_2").exists()


def test_get_ignored_relative_paths():
    ignore_patterns = [
        ".mypy_cache/",
        "venv/",
        "*.tmp",
        ".git",
        "data/*",
    ]

    root_relative_paths = {
        ".mypy_cache/should_ignore.json",
        "venv/bin/activate",
        "ignored_file.tmp",
        ".git",
        ".git/HEAD",
        "data/should_ignore.txt",
        "data.txt" "config.yaml",
        "model/model.py",
    }

    ignored_relative_paths = path.get_ignored_relative_paths(
        root_relative_paths, ignore_patterns
    )
    assert set(ignored_relative_paths) == {
        ".mypy_cache/should_ignore.json",
        "venv/bin/activate",
        "ignored_file.tmp",
        ".git",
        ".git/HEAD",
        "data/should_ignore.txt",
    }


def test_get_ignored_relative_paths_from_root(custom_model_truss_dir_with_hidden_files):
    ignore_patterns = [
        "__pycache__",
        ".DS_Store",
        ".git",
        "data/*",
    ]

    unignored_relative_paths = path.get_unignored_relative_paths_from_root(
        custom_model_truss_dir_with_hidden_files, ignore_patterns
    )
    unignored_relative_path_strs = set(
        (str(unignored_path) for unignored_path in unignored_relative_paths)
    )
    assert unignored_relative_path_strs == {
        "model",
        "model/model.py",
        "model/__init__.py",
        "data",
        "config.yaml",
        "packages",
    }

    all_relative_path_strs = set(
        str(path.relative_to(custom_model_truss_dir_with_hidden_files))
        for path in custom_model_truss_dir_with_hidden_files.glob("**/*")
    )
    ignored_relative_paths_strs = {
        "__pycache__",
        "__pycache__/test.cpython-311.pyc",
        ".DS_Store",
        ".git",
        ".git/.test_file",
        "data/test_file",
    }
    assert (
        all_relative_path_strs
        == ignored_relative_paths_strs | unignored_relative_path_strs
    )
