"""uv-specific pieces of `truss train exec`.

Everything that knows what uv is lives here, so adding another project type (pip,
poetry, ...) means adding a sibling module rather than editing the exec builder.
"""

import logging
from pathlib import Path
from typing import List

import tomlkit

logger = logging.getLogger(__name__)

# The official uv image is the same slim Python as the non-uv default, with uv
# preinstalled. (uv stopped publishing bookworm-slim variants for current versions,
# hence trixie-slim.)
UV_BASE_IMAGE = "ghcr.io/astral-sh/uv:0.12.6-python3.12-trixie-slim"

UV_LOCK_FILE = "uv.lock"
PYPROJECT_FILE = "pyproject.toml"

# Skip-if-present, so this is safe on an image that already has uv. Needs curl,
# which the CUDA base image has but a custom --image may not.
#
# The download is separate from running it because in `curl ... | sh` the pipeline's
# status is `sh`'s: a failed download would feed an empty script to a shell that
# exits 0, so the `&&` chain would continue and the job would fail later with an
# opaque "uv: not found".
UV_INSTALL_SCRIPT_PATH = "/tmp/uv-install.sh"
UV_INSTALL_STEPS = [
    "{ command -v uv >/dev/null 2>&1 || { "
    f"curl -LsSf https://astral.sh/uv/install.sh -o {UV_INSTALL_SCRIPT_PATH} && "
    f"sh {UV_INSTALL_SCRIPT_PATH}"
    " ; } ; }",
    'export PATH="$HOME/.local/bin:$PATH"',
]


def is_uv_project(source_dir: Path) -> bool:
    """Whether `source_dir` carries uv project metadata.

    A `pyproject.toml` on its own is not enough -- poetry, hatch, PDM and setuptools
    all ship one -- so require a uv lockfile or an explicit `[tool.uv]` section.
    """
    if (source_dir / UV_LOCK_FILE).exists():
        return True

    pyproject = source_dir / PYPROJECT_FILE
    if not pyproject.is_file():
        return False
    try:
        return "uv" in (tomlkit.parse(pyproject.read_text()).get("tool") or {})
    except Exception:
        logger.debug("Could not read %s for uv detection.", pyproject, exc_info=True)
        return False


class UvProject:
    """A project whose dependencies are managed by uv.

    Satisfies the `Project` protocol structurally; there is no `build()` because
    nothing is built locally -- the user's command does whatever building is needed.
    """

    label = "uv"

    def base_image(self) -> str:
        return UV_BASE_IMAGE

    def setup(self, base_image: str) -> List[str]:
        """Steps to run before the user's command, given the resolved base image.

        Empty on the uv image, which already ships uv. Anything else -- the CUDA
        image used for GPU jobs, or a custom --image we know nothing about -- gets
        the skip-if-present install.
        """
        if base_image == UV_BASE_IMAGE:
            return []
        return list(UV_INSTALL_STEPS)
