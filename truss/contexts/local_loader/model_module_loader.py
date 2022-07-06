import importlib
import sys
from contextlib import contextmanager
from importlib.machinery import PathFinder


class ModelModuleFinder(PathFinder):
    _truss_dir: str

    @classmethod
    def set_model_truss_dir(cls, truss_dir: str):
        cls._truss_dir = truss_dir
        cls.add_to_meta_path()

    @classmethod
    def find_spec(cls, fullname, path=None, target=None):
        if not path:
            path = [cls._truss_dir]
        else:
            path = list(path) + [cls._truss_dir]
        return PathFinder.find_spec(fullname, path, target)

    @classmethod
    def add_to_meta_path(cls):
        if cls not in sys.meta_path:
            sys.meta_path.append(cls)

    @classmethod
    def remove_from_meta_path(cls):
        if cls in sys.meta_path:
            del sys.meta_path[sys.meta_path.index(cls)]


class ModelModuleLoader:
    @staticmethod
    def import_model_module(name):
        """
        Custom import implementation for loading model module.

        Based on https://docs.python.org/3/library/importlib.html#approximating-importlib-import-module
        """
        absolute_name = importlib.util.resolve_name(name, None)
        path = None
        if '.' in absolute_name:
            parent_name, _, child_name = absolute_name.rpartition('.')
            parent_module = ModelModuleLoader.import_model_module(parent_name)
            path = parent_module.__spec__.submodule_search_locations

        spec = ModelModuleFinder.find_spec(absolute_name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[absolute_name] = module
        spec.loader.exec_module(module)
        if path is not None:
            setattr(parent_module, child_name, module)
        return module


@contextmanager
def model_class_module_loaded(truss_dir: str, model_class_module_fullname: str):
    try:
        model_module_name = model_class_module_fullname.split('.')[0]
        # Unload so that all model modules can be freshly loaded.
        _unload_model_modules(model_module_name)
        ModelModuleFinder.set_model_truss_dir(truss_dir)
        model_class_module = ModelModuleLoader.import_model_module(model_class_module_fullname)
        yield model_class_module
    finally:
        ModelModuleFinder.remove_from_meta_path()


def _unload_model_modules(model_module_name: str):
    model_module_submodules = [module_name
                               for module_name in sys.modules
                               if module_name.startswith(model_module_name + '.')]
    for model_module_submodule in model_module_submodules:
        del sys.modules[model_module_submodule]
