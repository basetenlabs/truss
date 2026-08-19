import importlib.util
import sys
from pathlib import Path


def setup_server_imports():
    """Add the server template directory to sys.path so that model_wrapper and
    _truss_common can be imported with the same relative-import layout they use
    at runtime.  Also registers truss/templates/shared as _truss_shared so that
    _truss_common sub-modules can import from it."""
    base_path = Path(__file__).parent.parent.parent.parent.parent

    server_path = base_path / "truss" / "templates" / "server"
    shared_path = base_path / "truss" / "templates" / "shared"

    for path in (server_path, shared_path):
        if not path.exists():
            raise FileNotFoundError(f"Expected path does not exist: {path}")

    path_str = str(server_path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

    if "_truss_shared" not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            "_truss_shared",
            str(shared_path / "__init__.py"),
            submodule_search_locations=[str(shared_path)],
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules["_truss_shared"] = mod
        spec.loader.exec_module(mod)
