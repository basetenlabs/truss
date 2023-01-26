from pathlib import Path

import yaml
from truss.local.local_config_handler import LocalConfigHandler
from truss.patch.hash import str_hash_str
from truss.truss_handle import TrussHandle
from truss.utils import copy_file_path, copy_tree_path, remove_tree_path


def gather(truss_path: Path) -> Path:
    handle = TrussHandle(truss_path)
    shadow_truss_dir_name = _calc_shadow_truss_dirname(truss_path)
    shadow_truss_metdata_file_path = (
        LocalConfigHandler.shadow_trusses_dir_path()
        / f"{shadow_truss_dir_name}.metadata.yaml"
    )
    shadow_truss_path = (
        LocalConfigHandler.shadow_trusses_dir_path() / shadow_truss_dir_name
    )
    if shadow_truss_metdata_file_path.exists():
        with shadow_truss_metdata_file_path.open() as fp:
            metadata = yaml.safe_load(fp)
        max_mod_time = metadata["max_mod_time"]
        if max_mod_time == handle.max_modified_time():
            return shadow_truss_path

        # Shadow truss is out of sync, clear it
        shadow_truss_metdata_file_path.unlink()
        remove_tree_path(shadow_truss_path)

    copy_tree_path(truss_path, shadow_truss_path)
    packages_dir_path_in_shadow = (
        shadow_truss_path / handle.spec.config.bundled_packages_dir
    )
    packages_dir_path_in_shadow.mkdir(exist_ok=True)
    for path in handle.spec.external_packages_dirs_paths:
        if not path.is_dir():
            raise ValueError(
                f"External packages directory at {path} is not a directory"
            )
        for sub_path in path.iterdir():
            if sub_path.is_dir():
                copy_tree_path(sub_path, packages_dir_path_in_shadow / sub_path.name)
            if sub_path.is_file():
                copy_file_path(sub_path, packages_dir_path_in_shadow / sub_path.name)

    shadow_handle = TrussHandle(shadow_truss_path)
    shadow_handle.clear_external_packages()
    with shadow_truss_metdata_file_path.open("w") as fp:
        yaml.safe_dump({"max_mod_time": handle.max_modified_time()}, fp)
    return shadow_truss_path


def _calc_shadow_truss_dirname(truss_path: Path) -> str:
    resolved_path_str = str(truss_path.resolve())
    return str_hash_str(resolved_path_str)
