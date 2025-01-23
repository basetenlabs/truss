from pathlib import Path
from typing import Any, List, Optional

from blake3 import blake3

from truss.util.path import get_unignored_relative_paths_from_root


def directory_content_hash(
    root: Path, ignore_patterns: Optional[List[str]] = None
) -> str:
    """Calculate content based hash of a filesystem directory.

    Rough algo: Sort all files by path, then take hash of a content stream, where
    we write path hash to the stream followed by hash of content if path is a file.
    Note the hash of hash aspect.

    Also, note that name of the root directory is not taken into account, only the contents
    underneath. The (root) Directory will have the same hash, even if renamed.
    """
    hasher = blake3()
    paths = list(get_unignored_relative_paths_from_root(root, ignore_patterns))
    paths.sort()
    for path in paths:
        hasher.update(str_hash(str(path)))
        absolute_path = root / path
        if absolute_path.is_file():
            hasher.update(file_content_hash(absolute_path))
    return hasher.hexdigest()


def file_content_hash(file: Path) -> bytes:
    """Calculate blake3 hash of file content.
    Returns: binary hash of content
    """
    return _file_content_hash_loaded_hasher(file).digest()


def file_content_hash_str(file: Path) -> str:
    """Calculate blake3 hash of file content.

    Returns: string hash of content
    """
    return _file_content_hash_loaded_hasher(file).hexdigest()


def _file_content_hash_loaded_hasher(file: Path) -> Any:
    hasher = blake3()
    buffer = bytearray(128 * 1024)
    mem_view = memoryview(buffer)
    with file.open("rb") as f:
        done = False
        while not done:
            n = f.readinto(mem_view)
            if n > 0:
                hasher.update(mem_view[:n])
            else:
                done = True
    return hasher


def str_hash(content: str) -> bytes:
    hasher = blake3()
    hasher.update(content.encode("utf-8"))
    return hasher.digest()


def str_hash_str(content: str) -> str:
    hasher = blake3()
    hasher.update(content.encode("utf-8"))
    return hasher.hexdigest()
