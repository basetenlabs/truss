import fnmatch
import os
import random
import string
import tempfile
from contextlib import contextmanager
from distutils.file_util import copy_file
from pathlib import Path
from shutil import copytree
from typing import List, Optional, Tuple

from truss.patch.hash import str_hash_str

# .truss_ignore is a fixed file in the Truss library that is used to specify files
# that should be ignored when copying a directory tree such as .git directory.
FIXED_TRUSS_IGNORE_PATH = Path(__file__).parent / ".truss_ignore"


def copy_tree_path(
    src: Path, dest: Path, directories_to_skip: Optional[List[str]] = None
) -> None:
    """Copy a directory tree, skipping certain top-level directories."""
    directories_to_skip = directories_to_skip or []

    def _ignore(src, names):
        return set(name for name in names if name in directories_to_skip)

    copytree(src, dest, ignore=_ignore, dirs_exist_ok=True)


def copy_file_path(src: Path, dest: Path) -> Tuple[str, str]:
    return copy_file(str(src), str(dest), verbose=False)


def copy_tree_or_file(src: Path, dest: Path) -> Optional[Tuple[str, str]]:
    if src.is_file():
        return copy_file_path(src, dest)

    return copy_tree_path(src, dest)  # type: ignore


def remove_tree_path(
    target: Path, directories_to_skip: Optional[List[str]] = None
) -> None:
    """Remove a directory tree, skipping certain top-level directories."""
    directories_to_skip = directories_to_skip or []

    for item in target.iterdir():
        if item.is_dir():
            if item.name not in directories_to_skip:
                remove_tree_path(item, directories_to_skip)
        else:
            item.unlink()

    if not any(target.iterdir()):
        target.rmdir()


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


def calc_shadow_truss_dirname(truss_path: Path) -> str:
    resolved_path_str = str(truss_path.resolve())
    return str_hash_str(resolved_path_str)


def build_truss_shadow_target_directory(stub: str, truss_path: Path) -> Path:
    """Builds a directory under ~/.truss/models."""
    suffix = calc_shadow_truss_dirname(truss_path)
    target_directory_path = Path(Path.home(), ".truss", "models", f"{stub}-{suffix}")
    target_directory_path.mkdir(parents=True, exist_ok=True)
    return target_directory_path


def load_trussignore_patterns(truss_ignore_file: Path):
    """Load patterns from a .truss_ignore file"""
    patterns = []
    with truss_ignore_file.open() as f:
        for line in f:
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
    if base_dir:
        path = path.relative_to(base_dir)

    while path:
        for pattern in patterns:
            if path.is_dir() and pattern.endswith("/"):
                pattern = pattern.rstrip("/")
                if fnmatch.fnmatch(path.name, pattern):
                    return True
            elif fnmatch.fnmatch(path.name, pattern):
                return True
        path = path.parent if path.parent != path else None  # type: ignore
    return False


def remove_ignored_files(
    directory: Path, truss_ignore_file: Path = FIXED_TRUSS_IGNORE_PATH
) -> None:
    """Traverse a directory and remove any files that match patterns in .truss_ignore"""
    patterns = load_trussignore_patterns(truss_ignore_file)
    for root, dirs, files in os.walk(directory, topdown=False):
        for name in files:
            file_path = Path(root) / name
            if is_ignored(file_path, patterns, base_dir=directory):
                file_path.unlink()
        for name in dirs:
            dir_path = Path(root) / name
            if is_ignored(dir_path, patterns, base_dir=directory):
                remove_tree_path(dir_path)


def files_in_dir_recursive(directory: Path) -> List[Path]:
    """Returns a list of all files in a directory, recursively."""
    return [
        file.relative_to(directory) for file in directory.glob("**/*") if file.is_file()
    ]


def are_dirs_equal(dir1: Path, dir2: Path) -> bool:
    """
    Checks if the contents of two directories are identical. This comparison includes the filenames,
    their sizes, and their modification times. This function doesn't check the actual content of the files.

    Args:
        dir1 (Path): The first directory to be compared.
        dir2 (Path): The second directory to be compared.

    Returns:
        bool: True if both directories contain the same set of files (names and sizes), and all corresponding
        files in both directories have the same modification time or the file in 'dir1' is older. If there are any
        discrepancies between the two directories (existence, file presence, file size, modification time),
        the function will return False.

    Note:
        This function does not check the content of the files. Two files are considered the same if they
        have the same name, size, and modification time, even if their contents are different.
    """

    if dir1.exists() != dir2.exists():
        return False

    files_in_first_directory = files_in_dir_recursive(dir1)
    files_in_second_directory = files_in_dir_recursive(dir2)

    if set(files_in_first_directory) != set(files_in_second_directory):
        return False

    for file in files_in_first_directory:
        f1_stat = os.stat(dir1 / file)
        f2_stat = os.stat(dir2 / file)
        len_f1, len_f2 = f1_stat.st_size, f2_stat.st_size
        mod_time_f1, mod_time_f2 = f1_stat.st_mtime, f2_stat.st_mtime

        if len_f1 != len_f2 or mod_time_f1 > mod_time_f2:
            return False

    return True
