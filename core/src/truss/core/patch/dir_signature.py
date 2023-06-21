from pathlib import Path
from typing import Dict

from truss.patch.hash import file_content_hash_str


def directory_content_signature(root: Path) -> Dict[str, str]:
    """Calculate content signature of a filesystem directory.

    Sort all files by path, store file path with content hash.
    Signatures are meant to track changes in directory. e.g.
    A previous signature and the directory can be combined to create
    a patch from the previous state.

    Hash of directories is marked None.
    """
    paths = list(root.glob("**/*"))
    paths.sort(key=lambda p: p.relative_to(root))

    def path_hash(pth: Path):
        if pth.is_file():
            return file_content_hash_str(pth)

    return {str(path.relative_to(root)): path_hash(path) for path in paths}
