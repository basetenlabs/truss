import importlib.metadata
import pathlib

import tomlkit


def _get_version() -> str:
    # For in-repo, `pyproject` takes precedence, but we need to assert it's
    # matching the truss package, because some other packages pollute `site-packages`
    # for pypi installs.
    toml_file = pathlib.Path(__file__).parent.parent / "pyproject.toml"
    if toml_file.exists():
        try:
            pyproject = tomlkit.parse(toml_file.read_text())
            poetry_section = pyproject["tool"]["poetry"]  # type: ignore[index]
            if poetry_section["name"] == __name__:  # type: ignore[index]
                return str(poetry_section["version"]).strip()  # type: ignore[index]
        except Exception:
            pass
    # Either there is no pyproject file or it's from a different package.
    # Try dist info metadata. This must be present.
    return importlib.metadata.version(__name__).strip()


__version__ = _get_version()


from truss.api import login, push, whoami
from truss.base import truss_config
from truss.truss_handle.build import load  # TODO: Refactor all usages and remove.

__all__ = ["push", "login", "load", "whoami", "truss_config"]
