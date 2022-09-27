import hashlib
from pathlib import Path


def directory_hash(root: Path) -> str:
    """Calculate content based hash of a filesystem directory.

    Rough algo: Sort all files by path, then take hash of a content stream, where
    we write path hash to the stream followed by hash of content if path is a file.
    Note the hash of hash aspect.
    """
    hasher = hashlib.sha256()
    hasher.update(root.name.encode("utf-8"))
    paths = [path for path in root.glob("**/*")]
    paths.sort(key=lambda p: p.relative_to(root))
    for path in paths:
        hasher.update(str_hash(str(path.relative_to(root))))
        if path.is_file():
            hasher.update(file_content_hash(path))
    return hasher.hexdigest()


def file_content_hash(file: Path):
    """Calculate sha256 of file content.

    [IMPORTANT] This function is copied directly to
                `build_model_from_truss_and_notify_task.yaml`.
                If this code is changed, then make sure to update that as well.
    """
    hasher = hashlib.sha256()
    buffer = bytearray(128 * 1024)
    mem_view = memoryview(buffer)
    with file.open("rb") as f:
        while n := f.readinto(mem_view):
            hasher.update(mem_view[:n])
    return hasher.digest()


def str_hash(content: str):
    hasher = hashlib.sha256()
    hasher.update(content.encode("utf-8"))
    return hasher.digest()
