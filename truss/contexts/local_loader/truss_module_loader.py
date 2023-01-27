import importlib
import sys
from contextlib import contextmanager
from importlib.machinery import PathFinder
from pathlib import Path
from typing import List


class TrussModuleFinder(PathFinder):
    _truss_dir: str
    _bundled_packages_dir_name: str
    _external_packages_dirs: List[str]

    @classmethod
    def set_model_truss_dirs(
        cls,
        truss_dir: str,
        truss_module_name: str = None,
        bundled_packages_dir_name: str = None,
        external_packages_dirs: List[str] = None,
    ):
        cls._truss_dir = truss_dir
        cls._truss_module_name = truss_module_name
        cls._bundled_packages_dir_name = bundled_packages_dir_name
        cls._external_packages_dirs = external_packages_dirs
        cls.add_to_meta_path()

    @classmethod
    def find_spec(cls, fullname, path=None, target=None):
        top_level_module_name = fullname.split(".")[0]
        if top_level_module_name == cls._truss_module_name:
            truss_modules_path = [cls._truss_dir]
        else:
            truss_modules_path = [
                str(Path(cls._truss_dir) / cls._bundled_packages_dir_name),
            ]
            if cls._external_packages_dirs:
                truss_modules_path += [
                    str(Path(cls._truss_dir) / d) for d in cls._external_packages_dirs
                ]
        if not path:
            path = truss_modules_path
        else:
            path = list(path) + truss_modules_path
        return PathFinder.find_spec(fullname, path, target)

    @classmethod
    def add_to_meta_path(cls):
        if cls not in sys.meta_path:
            sys.meta_path.append(cls)

    @classmethod
    def remove_from_meta_path(cls):
        if cls in sys.meta_path:
            del sys.meta_path[sys.meta_path.index(cls)]


class TrussModuleLoader:
    @staticmethod
    def import_truss_module(name):
        """
        Custom import implementation for loading model or training module.

        Based on https://docs.python.org/3/library/importlib.html#approximating-importlib-import-module
        """
        absolute_name = importlib.util.resolve_name(name, None)
        path = None
        if "." in absolute_name:
            parent_name, _, child_name = absolute_name.rpartition(".")
            parent_module = TrussModuleLoader.import_truss_module(parent_name)
            path = parent_module.__spec__.submodule_search_locations

        spec = TrussModuleFinder.find_spec(absolute_name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[absolute_name] = module
        spec.loader.exec_module(module)
        if path is not None:
            setattr(parent_module, child_name, module)
        return module


@contextmanager
def truss_module_loaded(
    truss_dir: str,
    truss_class_module_fullname: str,
    bundled_packages_dir_name: str = None,
    external_packages_dirs: List[str] = None,
):
    """
    Load a truss module: model or train.
    Args:
        truss_dir: Directory of truss
        truss_class_module_fullname: Full name of model or train module,
                                     e.g. 'model.model', 'train.train'
        bundled_packages_dir_name: name of the bundled packages directory
                                   if None then bundled packages are not loaded.
        external_packages_dirs: list of names of the external packages directories
                                   if None then external packages are not loaded.
    """
    try:
        truss_module_name = truss_class_module_fullname.split(".")[0]
        # Unload so that all modules can be freshly loaded.
        _unload_truss_modules(
            truss_dir,
            truss_module_name,
            bundled_packages_dir_name,
            external_packages_dirs,
        )
        TrussModuleFinder.set_model_truss_dirs(
            truss_dir,
            truss_module_name,
            bundled_packages_dir_name,
            external_packages_dirs,
        )
        class_module = TrussModuleLoader.import_truss_module(
            truss_class_module_fullname
        )
        yield class_module
    finally:
        TrussModuleFinder.remove_from_meta_path()


def _unload_truss_modules(
    truss_dir: str,
    truss_module_name: str,
    bundled_packages_dir_name: str = None,
    external_packages_dirs: List[str] = None,
):
    modules_to_unload = [truss_module_name]

    def _add_relative_dir_to_unload(dir_name: str):
        path = Path(truss_dir) / dir_name
        if path.exists():
            modules_to_unload.extend(_sub_dirnames(path))

    if bundled_packages_dir_name is not None:
        _add_relative_dir_to_unload(bundled_packages_dir_name)

    if external_packages_dirs is not None:
        for relative_dir in external_packages_dirs:
            _add_relative_dir_to_unload(relative_dir)
    _unload_top_level_modules(modules_to_unload)


def _unload_top_level_modules(module_names: List[str]):
    for module_name in module_names:
        _unload_top_level_module(module_name)


def _unload_top_level_module(module_name: str):
    if "." in module_name:
        raise ValueError(f"Expecting a top level module but found {module_name}")

    truss_module_submodules = [
        sys_module_name
        for sys_module_name in sys.modules
        if sys_module_name.startswith(module_name + ".")
    ]
    for truss_module_submodule in truss_module_submodules:
        del sys.modules[truss_module_submodule]
    if module_name in sys.modules:
        del sys.modules[module_name]


def _sub_dirnames(root_dir: Path):
    return [path.name for path in root_dir.iterdir() if path.is_dir()]
