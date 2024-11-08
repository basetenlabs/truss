import logging
from pathlib import Path
from typing import Dict, List, Optional, Set

import yaml
from truss.constants import CONFIG_FILE
from truss.patch.custom_types import ChangedPaths, TrussSignature
from truss.patch.hash import file_content_hash_str
from truss.templates.control.control.helpers.custom_types import (
    Action,
    ConfigPatch,
    DataPatch,
    EnvVarPatch,
    ExternalDataPatch,
    ModelCodePatch,
    PackagePatch,
    Patch,
    PatchType,
    PythonRequirementPatch,
    SystemPackagePatch,
)
from truss.templates.control.control.helpers.truss_patch.requirement_name_identifier import (
    RequirementMeta,
)
from truss.templates.control.control.helpers.truss_patch.system_packages import (
    system_packages_set,
)
from truss.truss_config import ExternalData, TrussConfig
from truss.truss_spec import TrussSpec
from truss.util.path import get_ignored_relative_paths

logger: logging.Logger = logging.getLogger(__name__)
PYCACHE_IGNORE_PATTERNS = [
    "**/__pycache__/**/*",
    "**/__pycache__/**",
]
UNPATCHABLE_CONFIG_KEYS = [
    "live_reload",
    "python_version",
    "resources",
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

    new_config = TrussConfig.from_yaml(truss_dir / CONFIG_FILE)
    prev_config = TrussConfig.from_dict(yaml.safe_load(previous_truss_signature.config))

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
        return _strictly_under(path, [data_dir_path])

    patches = []

    for path in changed_paths["removed"]:
        if _strictly_under(path, [model_module_path]):
            logger.info(f"Created patch to remove model code file: {path}")
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
        elif _strictly_under(path, [bundled_packages_path]):
            logger.info(f"Created patch to remove package file: {path}")
            patches.append(
                Patch(
                    type=PatchType.PACKAGE,
                    body=PackagePatch(
                        action=Action.REMOVE,
                        path=_relative_to(path, bundled_packages_path),
                    ),
                )
            )
        elif _under_unsupported_patch_dir(path):
            logger.warning(f"Patching not supported for removing {path}")
            return None

    has_calculated_config = False
    for path in changed_paths["added"] + changed_paths["updated"]:
        action = Action.ADD if path in changed_paths["added"] else Action.UPDATE
        if _strictly_under(path, [model_module_path]):
            full_path = truss_dir / path

            # TODO(pankaj) Add support for empty directories, skip them for now.
            if not full_path.is_file():
                continue
            logger.info(
                f"Created patch to {action.value.lower()} model code file: {path}"
            )
            patches.append(
                Patch(
                    type=PatchType.MODEL_CODE,
                    body=ModelCodePatch(
                        action=action,
                        path=_relative_to(path, model_module_path),
                        content=full_path.read_text(),
                    ),
                )
            )
        elif (
            path == CONFIG_FILE or _changed_path_is_requirements_file(path, new_config)
        ) and not has_calculated_config:
            # we could enter this code block from the requirements file's path
            # or from the config file's path. In any case, we only want to calculate these
            # patches once.
            has_calculated_config = True
            config_patches = calc_config_patches(
                truss_dir, previous_truss_signature, prev_config, new_config
            )
            if config_patches:
                logger.info(f"Created patch to {action.value.lower()} config")
            patches.extend(config_patches)
        elif _strictly_under(path, [bundled_packages_path]):
            full_path = truss_dir / path
            if not full_path.is_file():
                continue
            logger.info(f"Created patch to {action.value.lower()} package file: {path}")
            patches.append(
                Patch(
                    type=PatchType.PACKAGE,
                    body=PackagePatch(
                        action=action,
                        path=_relative_to(path, bundled_packages_path),
                        content=full_path.read_text(),
                    ),
                )
            )
        elif _under_unsupported_patch_dir(path):
            logger.info(f"Patching not supported for updating {path}")
            return None
    return patches


def _changed_path_is_requirements_file(changed_path: str, new_config: TrussConfig):
    """
    _changed_path_is_requirements_file determines if `changed_path` is the same path
    as the requirements file on a new configuration.
    """
    return new_config.requirements_file and Path(changed_path) == Path(
        new_config.requirements_file
    )


def _calc_changed_paths(
    root: Path,
    previous_root_path_content_hashes: Dict[str, str],
    ignore_patterns: Optional[List[str]],
) -> ChangedPaths:
    """
    TODO(pankaj) add support for directory creation in patch
    """
    root_relative_new_paths = set(
        (str(path.relative_to(root)) for path in root.glob("**/*"))
    )
    unignored_new_paths = calc_unignored_paths(root_relative_new_paths, ignore_patterns)
    previous_root_relative_paths = set(previous_root_path_content_hashes.keys())
    unignored_prev_paths = calc_unignored_paths(
        previous_root_relative_paths, ignore_patterns
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


def calc_unignored_paths(
    root_relative_paths: Set[str],
    ignore_patterns: Optional[List[str]] = None,
) -> Set[str]:
    ignored_paths = set(
        get_ignored_relative_paths(root_relative_paths, ignore_patterns)
    )
    return root_relative_paths - ignored_paths  # type: ignore


def calc_config_patches(
    truss_dir: Path,
    prev_signature: TrussSignature,
    prev_config: TrussConfig,
    new_config: TrussConfig,
) -> List[Patch]:
    """Calculate patch based on changes to config.

    Returns None if patch cannot be calculated. Empty list means no relevant
    differences found.
    """
    try:
        config_patches = _calc_general_config_patches(prev_config, new_config)
        python_requirements_patches = calc_requirements_patches(
            truss_dir, prev_signature, prev_config, new_config
        )
        system_package_patches = _calc_system_packages_patches(prev_config, new_config)
        return [*config_patches, *python_requirements_patches, *system_package_patches]
    except Exception as e:
        logger.error(f"Failed to calculate config patch with exception: {e}")
        raise


def _calc_general_config_patches(
    prev_config: TrussConfig, new_config: TrussConfig
) -> List[Patch]:
    """Calculate patch based on changes to config.yaml
    If a change has been made to the config, at least one ConfigPatch is created
    Additional patches are created for each of the patches that need specific application,
    namely patches to the env variables, external package dirs, and/or external data.

    Empty list means no relevant differences found.
    """

    patches: List[Patch] = []
    for key in UNPATCHABLE_CONFIG_KEYS:
        prev_items = getattr(prev_config, key)
        new_items = getattr(new_config, key)
        if new_items != prev_items:
            logger.warning(f"Patching is not supported for: {key}")

    if prev_config.to_dict() != new_config.to_dict():
        patches.append(_mk_config_patch(Action.UPDATE, new_config.to_dict()))

    env_var_patches = _calc_env_var_patches(prev_config, new_config)
    external_data_patches = _calc_external_data_patches(prev_config, new_config)

    return [*patches, *env_var_patches, *external_data_patches]


def _calc_env_var_patches(
    prev_config: TrussConfig, new_config: TrussConfig
) -> List[Patch]:
    """Calculate patch based on changes to environment variables.

    Empty list means no relevant differences found.
    """
    patches = []
    prev_items = prev_config.environment_variables
    new_items = new_config.environment_variables
    prev_item_names = set(prev_items.keys())
    new_item_names = set(new_items.keys())
    removed_items = prev_item_names.difference(new_item_names)
    for removed_item in removed_items:
        patches.append(
            _mk_env_var_patch(
                Action.REMOVE,
                {removed_item: prev_items[removed_item]},
            )
        )
    added_items = new_item_names.difference(prev_item_names)
    for added_item in added_items:
        patches.append(
            _mk_env_var_patch(Action.ADD, {added_item: new_items[added_item]})
        )
    for item in new_item_names.intersection(prev_item_names):
        if new_items[item] != prev_items[item]:
            patches.append(_mk_env_var_patch(Action.UPDATE, {item: new_items[item]}))
    return patches


def _calc_external_data_patches(
    prev_config: TrussConfig, new_config: TrussConfig
) -> List[Patch]:
    """Calculate patch based on changes to external data.

    Empty list means no relevant differences found.
    """
    patches = []
    prev_items = (prev_config.external_data or ExternalData([])).to_list()
    new_items = (new_config.external_data or ExternalData([])).to_list()

    removed_items = [x for x in prev_items if x not in new_items]
    for removed_item in removed_items:
        patches.append(
            _mk_external_data_patch(
                Action.REMOVE,
                removed_item,
            )
        )
    added_items = [x for x in new_items if x not in prev_items]
    for added_item in added_items:
        patches.append(_mk_external_data_patch(Action.ADD, added_item))
    return patches


def calc_requirements_patches(
    truss_dir: Path,
    prev_signature: TrussSignature,
    prev_config: TrussConfig,
    new_config: TrussConfig,
) -> List[Patch]:
    """
    requirements patches that are accounted for
    - contents in the requirements file change
    - requirements move from config file to requirements file
    - requirements move from requirements file to config file
    - requirement file changes (i.e. requirement.txt --> requirements.txt)
    - requirements in the config file change

    assumes that only one of requirements or requirements_file is present for a given config
    """
    prev_requirements = (
        prev_config.requirements
        if not prev_config.requirements_file
        else prev_signature.requirements_file_requirements
    )
    new_requirements = (
        new_config.requirements
        if not new_config.requirements_file
        else new_config.load_requirements_from_file(truss_dir)
    )
    return _calc_python_requirements_patches(prev_requirements, new_requirements)


def _calc_python_requirements_patches(
    prev_raw_reqs: List[str], new_raw_reqs: List[str]
) -> List[Patch]:
    """Calculate patch based on changes to python requirements.

    Empty list means no relevant differences found.
    """
    patches = []

    def create_requirement_map(raw_reqs: List[str]) -> Dict[str, RequirementMeta]:
        req_map = {}
        for raw_req in raw_reqs:
            meta = RequirementMeta.from_req(raw_req)
            req_map[meta.name] = meta
        return req_map

    prev_reqs_map = create_requirement_map(prev_raw_reqs)
    new_reqs_map = create_requirement_map(new_raw_reqs)
    prev_req_names = set(prev_reqs_map.keys())
    new_req_names = set(new_reqs_map.keys())

    removed_req_names = prev_req_names.difference(new_req_names)
    for removed_req_name in removed_req_names:
        removed_req_meta = prev_reqs_map[removed_req_name]
        if removed_req_meta.is_url_based_requirement:
            if not removed_req_meta.egg_tag:
                logger.warning(
                    f"Url-based requirement `{removed_req_meta.requirement}` is missing egg tag. Ignoring removal. Use `truss push` if you want to remove this requirement."
                )
                continue
            # make sure that the egg tag is included when removing the requirement
            patches.append(
                _mk_python_requirement_patch(
                    Action.REMOVE, removed_req_meta.requirement
                )
            )
        else:
            patches.append(
                _mk_python_requirement_patch(Action.REMOVE, removed_req_meta.name)
            )

    # warn for new reqs
    added_req_names = new_req_names.difference(prev_req_names)
    for added_req_name in added_req_names:
        added_req_meta = new_reqs_map[added_req_name]
        patches.append(
            _mk_python_requirement_patch(Action.ADD, added_req_meta.requirement)
        )
        if added_req_meta.is_url_based_requirement and not added_req_meta.egg_tag:
            logger.warning(
                f"Url-based requirement `{added_req_meta.requirement}` is missing egg tag. Removal will be ignored by `truss watch`"
            )
    for req_name in new_req_names.intersection(prev_req_names):
        if prev_reqs_map[req_name].requirement != new_reqs_map[req_name].requirement:
            patches.append(
                _mk_python_requirement_patch(
                    Action.UPDATE, new_reqs_map[req_name].requirement
                )
            )

    return patches


def _calc_system_packages_patches(
    prev_config: TrussConfig, new_config: TrussConfig
) -> List[Patch]:
    """Calculate patch based on changes to system packates.

    Empty list means no relevant differences found.
    """
    patches = []
    prev_pkgs = system_packages_set(prev_config.system_packages)
    new_pkgs = system_packages_set(new_config.system_packages)
    removed_pkgs = prev_pkgs.difference(new_pkgs)
    for removed_pkg in removed_pkgs:
        patches.append(_mk_system_package_patch(Action.REMOVE, removed_pkg))

    added_pkgs = new_pkgs.difference(prev_pkgs)
    for added_pkg in added_pkgs:
        patches.append(_mk_system_package_patch(Action.ADD, added_pkg))

    return patches


def _mk_config_patch(action: Action, config: dict) -> Patch:
    return Patch(
        type=PatchType.CONFIG,
        body=ConfigPatch(
            action=action,
            config=config,
        ),
    )


# Support for patching data changes yet to be implemented
def _mk_data_patch(action: Action, item: str, path: str) -> Patch:
    return Patch(
        type=PatchType.DATA,
        body=DataPatch(action=action, content=item, path=path),
    )


def _mk_env_var_patch(action: Action, item: dict) -> Patch:
    return Patch(
        type=PatchType.ENVIRONMENT_VARIABLE,
        body=EnvVarPatch(
            action=action,
            item=item,
        ),
    )


def _mk_external_data_patch(action: Action, item: Dict[str, str]) -> Patch:
    return Patch(
        type=PatchType.EXTERNAL_DATA,
        body=ExternalDataPatch(
            action=action,
            item=item,
        ),
    )


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
