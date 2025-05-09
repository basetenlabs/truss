from pathlib import Path
from typing import Optional

from truss import __version__

# This needs to be updated whenever we want to update
# base images on a merge. Updating this version will cause
# base images to be pushed with this tag.
TRUSS_BASE_IMAGE_VERSION_TAG = "v0.9.0"


def file_is_empty(path: Path, ignore_hash_style_comments: bool = True) -> bool:
    if not path.exists():
        return True

    with path.open() as file:
        for line in file.readlines():
            if ignore_hash_style_comments and _is_hash_style_comment(line):
                continue
            if line.strip() != "":
                return False

    return True


def file_is_not_empty(path: Path, ignore_hash_style_comments: bool = True) -> bool:
    return not file_is_empty(path, ignore_hash_style_comments)


def _is_hash_style_comment(line: str) -> bool:
    return line.lstrip().startswith("#")


def truss_base_image_tag(
    python_version: str, use_gpu: bool, version_tag: Optional[str] = None
) -> str:
    if version_tag is None:
        version_tag = f"v{__version__}"

    base_tag = python_version
    if use_gpu:
        base_tag = f"{base_tag}-gpu"
    return f"{base_tag}-{version_tag}"
