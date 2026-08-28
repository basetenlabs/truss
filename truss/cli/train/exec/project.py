"""Project-type detection for `truss train exec`.

A project type answers two questions about the directory being pushed: which base
image suits it, and what has to happen in the job before the user's command runs.
`get_project_type` is the single place that maps a directory to one, so supporting
pip or poetry later means adding a branch here plus a sibling module.
"""

from pathlib import Path
from typing import List, Optional, Protocol

from . import uv


class Project(Protocol):
    """A recognised project type in the directory `truss train exec` pushes."""

    #: Short label used in CLI messages, e.g. "uv".
    label: str

    def base_image(self) -> str:
        """The base image this project type wants when the user didn't pass --image."""
        ...

    def setup(self, base_image: str) -> List[str]:
        """Shell steps to run before the user's command, given the resolved image.

        Empty when the image already provides everything the project needs.
        """
        ...


def get_project_type(source_dir: Path) -> Optional[Project]:
    """The project type detected in `source_dir`, or None if none is recognised."""
    if uv.is_uv_project(source_dir):
        return uv.UvProject()
    return None
