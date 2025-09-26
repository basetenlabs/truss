import sys
from pathlib import Path


# NB(nikhil): Slightly hacky helpers needed to set up the path so relative imports work as they do in real environments
def setup_control_imports():
    base_path = Path(__file__).parent.parent.parent.parent.parent
    paths = [
        str(base_path / "templates" / "control" / "control"),
        str(base_path / "templates"),
        str(base_path / "templates" / "shared"),
    ]
    for path in paths:
        if path not in sys.path:
            sys.path.insert(0, path)
