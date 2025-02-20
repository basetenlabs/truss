import warnings
from pathlib import Path

from pydantic import PydanticDeprecatedSince20
from single_source import get_version

# Suppress Pydantic V1 warnings, because we have to use it for backwards compat.
warnings.filterwarnings("ignore", category=PydanticDeprecatedSince20)

__version__ = get_version(__name__, Path(__file__).parent.parent)


def version() -> str:
    return __version__ or ""


from truss.api import login, push, whoami
from truss.base import truss_config
from truss.truss_handle.build import load  # TODO: Refactor all usages and remove.

__all__ = ["push", "login", "load", "whoami", "truss_config"]
