import os
import random
import string
import subprocess as sp
import tempfile
from contextlib import contextmanager
from distutils.dir_util import copy_tree
from distutils.file_util import copy_file
from pathlib import Path


def copy_tree_path(src: Path, dest: Path):
    return copy_tree(str(src), str(dest))


def copy_file_path(src: Path, dest: Path):
    return copy_file(str(src), str(dest))


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
def given_or_temporary_dir(given_dir: Path = None):
    if given_dir is not None:
        yield given_dir
    else:
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)


def build_truss_target_directory(model_framework_name: str) -> Path:
    """Builds a directory under ~/.truss/models for the purpose of creating a Truss at."""
    rand_suffix = "".join(random.choices(string.ascii_uppercase + string.digits, k=6))
    target_directory_path = Path(
        Path.home(), ".truss", "models", f"{model_framework_name}-{rand_suffix}"
    )
    target_directory_path.mkdir(parents=True)
    return target_directory_path


def get_gpu_memory():
    # https://stackoverflow.com/questions/59567226/how-to-programmatically-determine-available-gpu-memory-with-tensorflow
    try:
        command = "nvidia-smi --query-gpu=memory.used --format=csv"
        memory_free_info = (
            sp.check_output(command.split()).decode("ascii").split("\n")[1]
        )
        memory_free_values = int(memory_free_info.split()[0])
        return memory_free_values
    except FileNotFoundError:
        return None
