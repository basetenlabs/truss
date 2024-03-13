from pathlib import Path
from typing import Dict, List, Optional

from truss.patch.hash import file_content_hash_str
from truss.util.path import get_unignored_relative_paths_from_root


def directory_content_signature(
    root: Path, ignore_patterns: Optional[List[str]] = None
) -> Dict[str, str]:
    """Calculate content signature of a filesystem directory.

    Sort all files by path, store file path with content hash.
    Signatures are meant to track changes in directory. e.g.
    A previous signature and the directory can be combined to create
    a patch from the previous state.

    Hash of directories is marked None.
    """
    paths = list(get_unignored_relative_paths_from_root(root, ignore_patterns))
    paths.sort()

    def path_hash(pth: Path):
        if pth.is_file():
            return file_content_hash_str(pth)

    return {str(path): path_hash(root / path) for path in paths}
