import warnings
from pathlib import Path

from pydantic import PydanticDeprecatedSince20
from single_source import get_version

# Suppress Pydantic V1 warnings, because we have to use it for backwards compat.
warnings.filterwarnings("ignore", category=PydanticDeprecatedSince20)


__version__ = get_version(__name__, Path(__file__).parent.parent)


def version():
    return __version__


from truss.api import login, push
from truss.build import from_directory, init, kill_all, load

__all__ = ["from_directory", "init", "kill_all", "load", "push", "login"]
