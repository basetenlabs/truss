from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import sys
import shlex

from .exceptions import InvalidError, NotFoundError


def validate_python_version(version: str) -> None:
    components = version.split(".")
    supported_versions = {"3.12", "3.11", "3.10", "3.9", "3.8"}
    if len(components) == 2 and version in supported_versions:
        return
    elif len(components) == 3:
        raise InvalidError(
            f"major.minor.patch version specification not valid. Supported major.minor versions are {supported_versions}."
        )
    raise InvalidError(
        f"Unsupported version {version}. Supported versions are {supported_versions}."
    )


def dockerhub_python_version(python_version=None):
    # TODO(bola): support typed truss config.
    # if python_version is None:
    #     python_version = config["image_python_version"]
    if python_version is None:
        python_version = "%d.%d" % sys.version_info[:2]

    parts = python_version.split(".")

    if len(parts) > 2:
        return python_version

    # We use the same major/minor version, but the highest micro version
    # See https://hub.docker.com/_/python
    latest_micro_version = {
        "3.12": "1",
        "3.11": "0",
        "3.10": "8",
        "3.9": "15",
        "3.8": "15",
    }
    major_minor_version = ".".join(parts[:2])
    python_version = (
        major_minor_version + "." + latest_micro_version[major_minor_version]
    )
    return python_version


def flatten_str_args(
    function_name: str, arg_name: str, args: Tuple[Union[str, List[str]], ...]
) -> List[str]:
    """Takes a tuple of strings, or string lists, and flattens it.

    Raises an error if any of the elements are not strings or string lists.
    """

    def is_str_list(x):
        return isinstance(x, list) and all(isinstance(y, str) for y in x)

    ret: List[str] = []
    for x in args:
        if isinstance(x, str):
            ret.append(x)
        elif is_str_list(x):
            ret.extend(x)
        else:
            raise InvalidError(f"{function_name}: {arg_name} must only contain strings")
    return ret


def make_pip_install_args(
    find_links: Optional[str] = None,  # Passes -f (--find-links) pip install
    index_url: Optional[str] = None,  # Passes -i (--index-url) to pip install
    extra_index_url: Optional[str] = None,  # Passes --extra-index-url to pip install
    pre: bool = False,  # Passes --pre (allow pre-releases) to pip install
) -> str:
    flags = [
        ("--find-links", find_links),
        ("--index-url", index_url),
        ("--extra-index-url", extra_index_url),
    ]

    args = " ".join(
        flag + " " + shlex.quote(value) for flag, value in flags if value is not None
    )
    if pre:
        args += " --pre"

    return args
