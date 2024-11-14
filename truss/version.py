from pathlib import Path

from single_source import get_version

VERSION = get_version(__name__, Path(__file__).parent.parent)
