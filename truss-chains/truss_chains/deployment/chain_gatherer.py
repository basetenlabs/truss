"""Chain gatherer for bundling external packages into a chain archive.

This module provides functionality to gather external_package_dirs from chainlets
into a self-contained chain archive, similar to how regular Truss handles
external packages via truss_handle/truss_gatherer.py.
"""

import pathlib
import tempfile

from truss.truss_handle.truss_handle import TrussHandle
from truss.util.path import copy_file_path, copy_tree_path

BUNDLED_PACKAGES_DIR = "packages"


def gather_chain(
    chain_root: pathlib.Path, external_package_dirs: list[pathlib.Path]
) -> pathlib.Path:
    """
    Creates a gathered version of the chain that includes all external_package_dirs.

    Args:
        chain_root: The root directory of the chain.
        external_package_dirs: List of external package directories to bundle.
            Caller is responsible for ensuring this list is non-empty and deduplicated.

    Returns:
        A new temp directory containing the chain with bundled packages.
    """
    # Create gathered chain in temp directory
    gathered_chain_root = pathlib.Path(tempfile.mkdtemp(prefix="gathered_chain_"))

    # Copy chain_root contents to gathered directory
    copy_tree_path(chain_root, gathered_chain_root)

    # Create packages directory for bundled external packages
    packages_dir = gathered_chain_root / BUNDLED_PACKAGES_DIR
    packages_dir.mkdir(exist_ok=True)

    for ext_dir in external_package_dirs:
        if not ext_dir.is_dir():
            raise ValueError(
                f"External packages directory at {ext_dir} is not a directory"
            )

        # Copy contents of the external package directory, not the directory itself.
        # This mimics the behavior in truss_gatherer.py and replicates adding
        # external package directory to sys.path.
        for sub_path in ext_dir.iterdir():
            dest_path = packages_dir / sub_path.name
            if sub_path.is_dir():
                copy_tree_path(sub_path, dest_path)
            elif sub_path.is_file():
                copy_file_path(sub_path, dest_path)

    # Clear external_package_dirs from all chainlet configs so the gathered chain
    # can be re-pushed without needing the original external paths.
    # This mirrors truss_gatherer.py's shadow_handle.clear_external_packages().
    _clear_external_package_dirs_from_configs(gathered_chain_root)

    return gathered_chain_root


def _clear_external_package_dirs_from_configs(chain_root: pathlib.Path) -> None:
    """Clear external_package_dirs from all chainlet config.yaml files.

    After gathering, external packages are bundled into packages/, so the
    original external_package_dirs paths are no longer needed. Clearing them
    allows the gathered chain to be downloaded and re-pushed without errors.

    Reuses TrussHandle.clear_external_packages() since each chainlet directory
    is a valid truss directory structure.
    """
    for chainlet_dir in chain_root.glob("chainlet_*"):
        if chainlet_dir.is_dir():
            handle = TrussHandle(chainlet_dir, validate=False)
            if handle.spec.config.external_package_dirs:
                handle.clear_external_packages()
