import sys
from pathlib import Path


# NB(nikhil): Slightly hacky helpers needed to set up the path so relative imports work as they do in real environments
def setup_control_imports():
    base_path = Path(__file__).parent.parent.parent.parent.parent
    paths = [
        base_path / "templates" / "control" / "control",
        base_path / "templates",
        base_path / "templates" / "shared",
    ]

    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Expected control path does not exist: {path}")

        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
