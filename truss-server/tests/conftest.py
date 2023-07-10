import contextlib
import os
import sys
from pathlib import Path

import pytest
from truss.core.util.path import copy_tree_path


@pytest.fixture
def truss_container_fs(tmp_path):
    ROOT = Path(__file__).parent.parent.parent.resolve()
    truss_dir = ROOT / "truss" / "tests" / "test_data" / "test_truss"
    tmp_truss_dir = tmp_path / "tmp_truss"
    tmp_truss_dir.mkdir()
    copy_tree_path(truss_dir, tmp_truss_dir)

    return tmp_truss_dir


class Helpers:
    @staticmethod
    @contextlib.contextmanager
    def file_content(file_path: Path, content: str):
        orig_content = file_path.read_text()
        try:
            file_path.write_text(content)
            yield
        finally:
            file_path.write_text(orig_content)

    @staticmethod
    @contextlib.contextmanager
    def sys_path(path: Path):
        try:
            sys.path.append(str(path))
            yield
        finally:
            sys.path.pop()

    @staticmethod
    @contextlib.contextmanager
    def env_var(var: str, value: str):
        orig_environ = os.environ.copy()
        try:
            os.environ[var] = value
            yield
        finally:
            os.environ.clear()
            os.environ.update(orig_environ)


@pytest.fixture
def helpers():
    return Helpers()
