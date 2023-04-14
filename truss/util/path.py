import os
import random
import string
import tempfile
from contextlib import contextmanager
from distutils.dir_util import copy_tree, remove_tree
from distutils.file_util import copy_file
from pathlib import Path
from typing import List, Optional, Tuple, Union


def copy_tree_path(src: Path, dest: Path) -> List[str]:
    return copy_tree(str(src), str(dest))


def copy_file_path(src: Path, dest: Path) -> Tuple[str, str]:
    return copy_file(str(src), str(dest))


def copy_tree_or_file(src: Path, dest: Path) -> Union[List[str], Tuple[str, str]]:
    if src.is_file():
        return copy_file_path(src, dest)

    return copy_tree_path(src, dest)


def remove_tree_path(target: Path) -> None:
    return remove_tree(str(target))


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
