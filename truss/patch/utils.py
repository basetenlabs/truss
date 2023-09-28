import fnmatch
from pathlib import Path
from typing import List, Optional


def path_matches_any_pattern(path: Path, patterns: Optional[List[str]]) -> bool:
    if patterns is None:
        return False

    for pattern in patterns:
        if fnmatch.fnmatch(str(path), pattern):
            return True

    return False
