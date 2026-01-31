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

    If no external_package_dirs, returns chain_root unchanged.
    Otherwise, creates a temp directory with:
    - All contents of chain_root
    - External packages bundled into packages/

    This mimics the behavior of truss_handle/truss_gatherer.py:gather() for
    regular Truss, ensuring that downloaded chain artifacts include all
    external dependencies.
    """
    if not external_package_dirs:
        return chain_root

    # Filter out any external dirs that are already inside chain_root
    # and deduplicate paths
    external_dirs_to_bundle: list[pathlib.Path] = []
    seen_paths: set[pathlib.Path] = set()
    chain_root_resolved = chain_root.resolve()
    for ext_dir in external_package_dirs:
        ext_dir_resolved = ext_dir.resolve()

        # Skip duplicates
        if ext_dir_resolved in seen_paths:
            continue
        seen_paths.add(ext_dir_resolved)

        try:
            ext_dir_resolved.relative_to(chain_root_resolved)
            # If we get here, ext_dir is inside chain_root, skip it
        except ValueError:
            # ext_dir is outside chain_root, need to bundle it
            external_dirs_to_bundle.append(ext_dir_resolved)

    if not external_dirs_to_bundle:
        return chain_root

    # Create gathered chain in temp directory
    gathered_chain_root = pathlib.Path(tempfile.mkdtemp(prefix="gathered_chain_"))

    # Copy chain_root contents to gathered directory
    copy_tree_path(chain_root, gathered_chain_root)

    # Create packages directory for bundled external packages
    packages_dir = gathered_chain_root / BUNDLED_PACKAGES_DIR
    packages_dir.mkdir(exist_ok=True)

    # Track used names to handle conflicts
    used_names: dict[str, int] = {}

    for ext_dir in external_dirs_to_bundle:
        if not ext_dir.is_dir():
            raise ValueError(
                f"External packages directory at {ext_dir} is not a directory"
            )

        # Copy contents of the external package directory, not the directory itself.
        # This mimics the behavior in truss_gatherer.py and replicates adding
        # external package directory to sys.path.
        for sub_path in ext_dir.iterdir():
            dest_name = sub_path.name

            # Handle name conflicts by adding a suffix
            if dest_name in used_names:
                used_names[dest_name] += 1
                name_stem = sub_path.stem
                name_suffix = sub_path.suffix
                dest_name = f"{name_stem}_{used_names[dest_name]}{name_suffix}"
            else:
                used_names[dest_name] = 0

            dest_path = packages_dir / dest_name

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
