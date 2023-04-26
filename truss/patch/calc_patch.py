import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pkg_resources
import yaml
from truss.constants import CONFIG_FILE
from truss.patch.hash import file_content_hash_str
from truss.patch.types import TrussSignature
from truss.templates.control.control.helpers.types import (
    Action,
    ModelCodePatch,
    Patch,
    PatchType,
    PythonRequirementPatch,
    SystemPackagePatch,
)
from truss.truss_config import TrussConfig
from truss.truss_spec import TrussSpec

logger: logging.Logger = logging.getLogger(__name__)
PYCACHE_IGNORE_PATTERNS = [
    "**/__pycache__/**/*",
    "**/__pycache__/**",
]


def calc_truss_patch(
    truss_dir: Path,
    previous_truss_signature: TrussSignature,
    ignore_patterns: Optional[List[str]] = None,
) -> Optional[List[Patch]]:
    """
    Calculate patch for a truss from a previous state.

    Returns: None if patch cannot be calculated, otherwise a list of patches.
        Note that the none return value is pretty important, patch coverage is
        limited and this usually indicates that the identified change cannot be
        expressed with currently supported patches.

        Only standard and relevant truss paths are scanned for changes, rest
        ignored. E.g. at the root level, only changes to config.yaml are
        checked, any other changes are ignored.
    """

    def _relative_to_root(path: Path) -> str:
        return str(path.relative_to(truss_dir))

    if ignore_patterns is None:
        ignore_patterns = PYCACHE_IGNORE_PATTERNS

    changed_paths = _calc_changed_paths(
        truss_dir,
        previous_truss_signature.content_hashes_by_path,
        ignore_patterns,
    )
    # TODO(pankaj) Calculate model code patches only for now, add config changes
    # later.

    truss_spec = TrussSpec(truss_dir)
    model_module_path = _relative_to_root(truss_spec.model_module_dir)
    data_dir_path = _relative_to_root(truss_spec.data_dir)
    bundled_packages_path = _relative_to_root(truss_spec.bundled_packages_dir)

    def _under_unsupported_patch_dir(path: str) -> bool:
        """
        Checks if the given path is under one of the directories that don't
        support patching. Note that if path `is` one of those directories that's
        ok, because those empty directories can be ignored from patching point
        of view.
        """
        return _strictly_under(path, [data_dir_path, bundled_packages_path])

    patches = []
    for path in changed_paths["removed"]:
        if path.startswith(model_module_path):
            patches.append(
                Patch(
                    type=PatchType.MODEL_CODE,
                    body=ModelCodePatch(
                        action=Action.REMOVE,
                        path=_relative_to(path, model_module_path),
                    ),
                )
            )
        elif path == CONFIG_FILE:
            # Don't support removal of config file
            logger.info(f"Patching not supported for removing {path}")
            return None
        elif _under_unsupported_patch_dir(path):
            logger.info(f"Patching not supported for removing {path}")
            return None

    for path in changed_paths["added"] + changed_paths["updated"]:
        if path.startswith(model_module_path):
            full_path = truss_dir / path

            # TODO(pankaj) Add support for empty directories, skip them for now.
            if not full_path.is_file():
                continue

            action = Action.ADD if path in changed_paths["added"] else Action.UPDATE
            patches.append(
                Patch(
                    type=PatchType.MODEL_CODE,
                    body=ModelCodePatch(
                        action=action,
                        path=_relative_to(path, model_module_path),
                        content=_file_content(full_path),
                    ),
                )
            )
        elif path == CONFIG_FILE:
            new_config = TrussConfig.from_yaml(truss_dir / CONFIG_FILE)
            prev_config = TrussConfig.from_dict(
                yaml.safe_load(previous_truss_signature.config)
            )
            config_patches = _calc_config_patches(prev_config, new_config)
            if config_patches is None:
                logger.info(f"Unable to patch update to {path}")
                return None
            patches.extend(config_patches)
        elif _under_unsupported_patch_dir(path):
            logger.info(f"Patching not supported for updating {path}")
            return None
    return patches


def _calc_changed_paths(
    root: Path,
    previous_root_path_content_hashes: Dict[str, str],
    ignore_patterns: Optional[List[str]],
) -> Dict[str, List[str]]:
    """
    TODO(pankaj) add support for directory creation in patch
    """
    root_relative_new_paths = set(
        (str(path.relative_to(root)) for path in root.glob("**/*"))
    )
    unignored_new_paths = _calc_unignored_paths(
        root, root_relative_new_paths, ignore_patterns
    )
    previous_root_relative_paths = set(previous_root_path_content_hashes.keys())
    unignored_prev_paths = _calc_unignored_paths(
        root, previous_root_relative_paths, ignore_patterns
    )

    added_paths = unignored_new_paths - unignored_prev_paths
    removed_paths = unignored_prev_paths - unignored_new_paths

    updated_paths = set()
    common_paths = unignored_new_paths.intersection(unignored_prev_paths)
    for path in common_paths:
        full_path: Path = root / path
        if full_path.is_file():
            content_hash = file_content_hash_str(full_path)
            previous_content_hash = previous_root_path_content_hashes[path]
            if content_hash != previous_content_hash:
                updated_paths.add(path)

    return {
        "added": list(added_paths),
        "updated": list(updated_paths),
        "removed": list(removed_paths),
    }


def _calc_unignored_paths(
    root: Path,
    root_relative_paths: Set[str],
    ignore_patterns: Optional[List[str]] = None,
) -> Set[str]:
    root_relative_ignored_paths = set()
    if ignore_patterns is not None:
        for ignore_pattern in ignore_patterns:
            ignored_paths_for_pattern = set(
                (str(path.relative_to(root)) for path in root.glob(ignore_pattern))
            )
            root_relative_ignored_paths.update(ignored_paths_for_pattern)

    return root_relative_paths - root_relative_ignored_paths


def _calc_config_patches(
    prev_config: TrussConfig, new_config: TrussConfig
) -> Optional[List[Patch]]:
    """Calculate patch based on changes to config.

    Returns None if patch cannot be calculated. Empty list means no relevant
    differences found.
    """
    # Currently only calculate patches for python requirements and system
    # packages, bail out if anything else has changed.
    if not _only_expected_config_differences(prev_config, new_config):
        return None

    python_requirement_patches = _calc_python_requirements_patches(
        prev_config, new_config
    )
    system_package_patches = _calc_system_packages_patches(prev_config, new_config)
    return [*python_requirement_patches, *system_package_patches]


def _calc_python_requirements_patches(
    prev_config: TrussConfig, new_config: TrussConfig
) -> List[Patch]:
    """Calculate patch based on changes to python requirements.

    Returns None if patch cannot be calculated. Empty list means no relevant
    differences found.
    """
    patches = []
    prev_reqs = _parsed_reqs_by_name(prev_config.requirements)
    prev_req_names = set(prev_reqs.keys())
    new_reqs = _parsed_reqs_by_name(new_config.requirements)
    new_req_names = set(new_reqs.keys())
    removed_reqs = prev_req_names.difference(new_req_names)
    for removed_req in removed_reqs:
        patches.append(_mk_python_requirement_patch(Action.REMOVE, removed_req))

    added_reqs = new_req_names.difference(prev_req_names)
    for added_req in added_reqs:
        patches.append(
            _mk_python_requirement_patch(Action.ADD, str(new_reqs[added_req]))
        )

    for req in new_req_names.intersection(prev_req_names):
        if new_reqs[req] != prev_reqs[req]:
            patches.append(
                _mk_python_requirement_patch(Action.UPDATE, str(new_reqs[req]))
            )

    return patches


def _calc_system_packages_patches(
    prev_config: TrussConfig, new_config: TrussConfig
) -> List[Patch]:
    """Calculate patch based on changes to system packates.

    Returns None if patch cannot be calculated. Empty list means no relevant
    differences found.
    """
    patches = []
    prev_pkgs = _system_pacakges_set(prev_config)
    new_pkgs = _system_pacakges_set(new_config)
    removed_pkgs = prev_pkgs.difference(new_pkgs)
    for removed_pkg in removed_pkgs:
        patches.append(_mk_system_package_patch(Action.REMOVE, removed_pkg))

    added_pkgs = new_pkgs.difference(prev_pkgs)
    for added_pkg in added_pkgs:
        patches.append(_mk_system_package_patch(Action.ADD, added_pkg))

    return patches


def _system_pacakges_set(config: TrussConfig) -> Set[str]:
    pkgs = []
    for sys_pkg_line in config.system_packages:
        pkgs.extend(sys_pkg_line.strip().split())
    return set(pkgs)


def _parsed_reqs_by_name(reqs: List[str]) -> Dict[str, Any]:
    parsed_reqs_by_name = {}
    for req in reqs:
        parsed_req = pkg_resources.Requirement.parse(req)
        parsed_reqs_by_name[parsed_req.name] = parsed_req  # type: ignore
    return parsed_reqs_by_name


def _only_expected_config_differences(
    prev_config: TrussConfig, new_config: TrussConfig
) -> bool:
    prev_config_dict = prev_config.to_dict()
    prev_config_dict["requirements"] = []
    prev_config_dict["system_packages"] = []

    new_config_dict = new_config.to_dict()
    new_config_dict["requirements"] = []
    new_config_dict["system_packages"] = []

    return prev_config_dict == new_config_dict


def _mk_python_requirement_patch(action: Action, requirement: str) -> Patch:
    return Patch(
        type=PatchType.PYTHON_REQUIREMENT,
        body=PythonRequirementPatch(
            action=action,
            requirement=requirement,
        ),
    )


def _mk_system_package_patch(action: Action, package: str) -> Patch:
    return Patch(
        type=PatchType.SYSTEM_PACKAGE,
        body=SystemPackagePatch(
            action=action,
            package=package,
        ),
    )


def _relative_to(path: str, relative_to_path: str):
    return str(Path(path).relative_to(relative_to_path))


def _strictly_under(path: str, parent_paths: List[str]) -> bool:
    """
    Checks if given path is under one of the given paths, but not the same as
    them. Assumes that parent paths themselves are not under each other.
    """
    for dir_path in parent_paths:
        if path.startswith(dir_path) and not path == dir_path:
            return True
    return False


def _file_content(path: Path) -> str:
    with path.open() as file:
        return file.read()
