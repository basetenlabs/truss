from pathlib import Path

from truss import __version__

# This needs to be updated whenever new truss base images are published. This
# manual nature is by design; at this point we want any new base image releases
# to be reviewed via code change before landing. This value should be typically
# set to the latest version of truss library.
#
# [IMPORTANT] Make sure all images for this version are published to dockerhub
# before change to this value lands. This value is used to look for base images
# when building docker image for a truss.
TRUSS_BASE_IMAGE_VERSION_TAG = "v0.3.3"


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


def truss_base_image_name(job_type: str) -> str:
    return f"baseten/truss-{job_type}-base"


def truss_base_image_tag(
    python_version: str,
    use_gpu: bool,
    live_reload: bool,
    version_tag: str = None,
) -> str:
    if version_tag is None:
        version_tag = f"v{__version__}"

    base_tag = python_version
    if use_gpu:
        base_tag = f"{base_tag}-gpu"
    if live_reload:
        base_tag = f"{base_tag}-reload"
    return f"{base_tag}-{version_tag}"


def to_dotted_python_version(truss_python_version: str) -> str:
    """Converts python version string using in truss config to the conventional dotted form.

    e.g. py39 to 3.9
    """
    return f"{truss_python_version[2]}.{truss_python_version[3:]}"
