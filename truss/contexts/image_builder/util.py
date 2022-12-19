from pathlib import Path

# This needs to be updated whenever new truss base images are published
TRUSS_BASE_IMAGE_TAG = "v0.1.11rc2"


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


def truss_base_image_name(
    job_type: str,
    python_version: str,
    use_gpu: bool,
    live_reload: bool,
) -> str:
    base_image_name = f"baseten/truss-{job_type}-base-{python_version}"
    if use_gpu:
        base_image_name = f"{base_image_name}-gpu"
    if live_reload:
        base_image_name = f"{base_image_name}-reload"
    return base_image_name


def to_dotted_python_version(truss_python_version: str) -> str:
    """Converts python version string using in truss config to the conventional dotted form.

    e.g. py39 to 3.9
    """
    return f"{truss_python_version[2]}.{truss_python_version[3:]}"
