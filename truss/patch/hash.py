import hashlib
from pathlib import Path


def directory_content_hash(root: Path) -> str:
    """Calculate content based hash of a filesystem directory.

    Rough algo: Sort all files by path, then take hash of a content stream, where
    we write path hash to the stream followed by hash of content if path is a file.
    Note the hash of hash aspect.

    Also, note that name of the root directory is not taken into account, only the contents
    underneath. The (root) Directory will have the same hash, even if renamed.
    """
    hasher = hashlib.sha256()
    paths = [path for path in root.glob("**/*")]
    paths.sort(key=lambda p: p.relative_to(root))
    for path in paths:
        hasher.update(str_hash(str(path.relative_to(root))))
        if path.is_file():
            hasher.update(file_content_hash(path))
    return hasher.hexdigest()


def file_content_hash(file: Path):
    """Calculate sha256 of file content.
    Returns: binary hash of content
    """
    return _file_content_hash_loaded_hasher(file).digest()


def file_content_hash_str(file: Path) -> str:
    """Calculate sha256 of file content.

    Returns: string hash of content
    """
    return _file_content_hash_loaded_hasher(file).hexdigest()


def _file_content_hash_loaded_hasher(file: Path):
    hasher = hashlib.sha256()
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


def str_hash(content: str):
    hasher = hashlib.sha256()
    hasher.update(content.encode("utf-8"))
    return hasher.digest()
