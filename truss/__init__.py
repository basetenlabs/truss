# flake8: noqa F401

from pathlib import Path

from single_source import get_version

__version__ = get_version(__name__, Path(__file__).parent.parent)

from truss.build import from_directory, init, kill_all, mk_truss
