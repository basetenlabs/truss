from pathlib import Path

from single_source import get_version

__version__ = get_version(__name__, Path(__file__).parent.parent)


def version():
    return __version__


from truss.build import from_directory, init, kill_all, load

__all__ = ["from_directory", "init", "kill_all", "load"]
