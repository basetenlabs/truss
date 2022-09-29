import os
from contextlib import contextmanager


@contextmanager
def current_directory(directory: str):
    """
    Execute with using given directory as current working directory.

    This can be problematic in a multi-threaded scenario. It's assumed that
    the caller will take locks as needed.
    """
    cwd = os.getcwd()
    os.chdir(directory)
    try:
        yield
    finally:
        os.chdir(cwd)
