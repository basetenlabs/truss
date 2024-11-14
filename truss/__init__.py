import pkgutil

__path__ = pkgutil.extend_path(__path__, __name__)

import warnings
from pathlib import Path

from pydantic import PydanticDeprecatedSince20
from single_source import get_version

# Suppress Pydantic V1 warnings, because we have to use it for backwards compat.
warnings.filterwarnings("ignore", category=PydanticDeprecatedSince20)

__version__ = get_version(__name__, Path(__file__).parent.parent)


def version():
    return __version__
