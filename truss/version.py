from pathlib import Path

from single_source import get_version

VERSION = get_version("truss_lib", Path(__file__).parent.parent, fail=True)
