import os
import random
import shutil
import string
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Set, Tuple, Union

import pathspec

# .truss_ignore is a fixed file in the Truss library that is used to specify files
# that should be ignored when copying a directory tree such as .git directory.
TRUSS_IGNORE_FILENAME = ".truss_ignore"
FIXED_TRUSS_IGNORE_PATH = Path(__file__).parent / TRUSS_IGNORE_FILENAME


def copy_tree_path(src: Path, dest: Path, ignore_patterns: List[str] = []) -> None:
    """Copy a directory tree, ignoring files specified in .truss_ignore."""
    patterns = load_trussignore_patterns()
    patterns.extend(ignore_patterns)

    if not dest.exists():
        dest.mkdir(parents=True)

    for sub_path in src.rglob("*"):
        if is_ignored(sub_path, patterns, base_dir=src):
            continue

        dest_fp = dest / sub_path.relative_to(src)

        if sub_path.is_dir():
            dest_fp.mkdir(exist_ok=True)
        else:
            dest_fp.parent.mkdir(exist_ok=True, parents=True)
            shutil.copy2(str(sub_path), str(dest_fp))


def copy_file_path(src: Path, dest: Path) -> Tuple[str, str]:
    return shutil.copy2(str(src), str(dest))


def copy_tree_or_file(src: Path, dest: Path) -> Union[List[str], Tuple[str, str]]:
    if src.is_file():
        return copy_file_path(src, dest)

    return copy_tree_path(src, dest)  # type: ignore


def remove_tree_path(target: Path) -> None:
    return shutil.rmtree(str(target))


def get_max_modified_time_of_dir(path: Path) -> float:
    max_modified_time = os.path.getmtime(path)
    for root, dirs, files in os.walk(path):
        if os.path.islink(root):
            raise ValueError(f"Symlinks not allowed in Truss: {root}")
        files = [f for f in files if not f.startswith(".")]
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        max_modified_time = max(max_modified_time, os.path.getmtime(root))
        for file in files:
            max_modified_time = max(
                max_modified_time, os.path.getmtime(os.path.join(root, file))
            )
    return max_modified_time


@contextmanager
def given_or_temporary_dir(given_dir: Optional[Path] = None):
    if given_dir is not None:
        yield given_dir
    else:
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)


def build_truss_target_directory(stub: str) -> Path:
    """Builds a directory under ~/.truss/models for the purpose of creating a Truss at."""
    rand_suffix = "".join(random.choices(string.ascii_uppercase + string.digits, k=6))
    target_directory_path = Path(
        Path.home(), ".truss", "models", f"{stub}-{rand_suffix}"
    )
    target_directory_path.mkdir(parents=True)
    return target_directory_path


def load_trussignore_patterns_from_truss_dir(truss_dir: Path) -> List[str]:
    truss_ignore_file = truss_dir / TRUSS_IGNORE_FILENAME
    if truss_ignore_file.exists():
        return load_trussignore_patterns(truss_ignore_file)
    # default to the truss-defined ignore patterns
    return load_trussignore_patterns()


def load_trussignore_patterns(
    truss_ignore_file: Path = FIXED_TRUSS_IGNORE_PATH,
) -> List[str]:
    """Load patterns from a .truss_ignore file"""
    patterns = []

    with truss_ignore_file.open() as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if line and not line.startswith("#"):
            patterns.append(line)

    return patterns


def is_ignored(
    path: Path, patterns: List[str], base_dir: Optional[Path] = None
) -> bool:
    """
    Determines if a specified path (or any of its parent directories) should be ignored
    based on a list of ignore patterns, analogous to how .gitignore works in Git. The
    function evaluates not only the path itself but all its parent directories up to
    the base directory. Hence, if a file resides within a directory to be ignored, it
    will be ignored as well. We provide the 'base_dir' argument to allow the caller to
    specify the base directory against which the relative 'path' is resolved. This
    prevents false positives where the absolute path before the base directory matches
    one of the ignore patterns.

    Args:
        path (Path): The path to the file or directory being checked. This can be absolute
            or relative. If 'base_dir' is provided, 'path' is considered relative to 'base_dir'.
        patterns (List[str]): A list of ignore patterns. Each pattern is a string that can contain
            wildcard characters. For instance, '*.py' would match all Python files, while 'test/'
            would match a directory named 'test'. A pattern ending with a slash (/) will only match directories.
        base_dir (Optional[Path]): If specified, it's used as the base directory against which
            the relative 'path' is resolved. If None, the 'path' is treated as either an absolute path
            or relative to the current working directory.

    Returns:
        bool: True if the path matches any of the ignore patterns (i.e., should be ignored),
            and False otherwise.
    """
    ignore_spec = pathspec.PathSpec.from_lines(
        pathspec.patterns.GitWildMatchPattern, patterns
    )

    if base_dir:
        path = path.relative_to(base_dir)

    return ignore_spec.match_file(path)


def get_ignored_relative_paths(
    root_relative_paths: Iterable[Union[str, os.PathLike]],
    ignore_patterns: Optional[List[str]] = None,
) -> Iterator[Union[str, os.PathLike]]:
    """Given an iterable of relative paths, returns an iterator of the relative paths that match ignore_patterns."""
    if ignore_patterns is None:
        return iter([])

    ignore_spec = pathspec.PathSpec.from_lines(
        pathspec.patterns.GitWildMatchPattern, ignore_patterns
    )
    return ignore_spec.match_files(root_relative_paths)


def get_unignored_relative_paths_from_root(
    root: Path, ignore_patterns: Optional[List[str]] = None
) -> Set[Path]:
    """Given a root directory, returns an iterator of the relative paths that do not match ignore_patterns."""
    root_relative_paths = set(path.relative_to(root) for path in root.glob("**/*"))

    ignored_paths = set(
        get_ignored_relative_paths(root_relative_paths, ignore_patterns)
    )
    return root_relative_paths - ignored_paths  # type: ignore
