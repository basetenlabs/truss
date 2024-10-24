import logging
import sys

logger: logging.Logger = logging.getLogger(__name__)


def infer_python_version() -> str:
    return f"py{sys.version_info.major}{sys.version_info.minor}"


def map_to_supported_python_version(python_version: str) -> str:
    """Map python version to truss supported python version.

    Currently, it maps any versions greater than 3.11 to 3.11.

    Args:
        python_version: in the form py[major_version][minor_version] e.g. py39,
        py310
    """
    python_major_version = int(python_version[2:3])
    python_minor_version = int(python_version[3:])

    if python_major_version != 3:
        raise NotImplementedError("Only python version 3 is supported")

    if python_minor_version > 11:
        logger.info(
            f"Mapping python version {python_major_version}.{python_minor_version}"
            " to 3.11, the highest version that Truss currently supports."
        )
        return "py311"

    if python_minor_version < 8:
        # TODO: consider raising an error instead - it doesn't' seem safe.
        logger.info(
            f"Mapping python version {python_major_version}.{python_minor_version}"
            " to 3.8, the lowest version that Truss currently supports."
        )
        return "py38"

    return python_version
