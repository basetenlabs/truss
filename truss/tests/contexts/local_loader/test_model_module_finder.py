import tempfile
from pathlib import Path

from truss.contexts.local_loader.model_module_loader import \
    model_class_module_loaded

ORIG_MODEL_CLASS_CONTENT = '''
class Model:
    x = 1
'''

NEW_MODEL_CLASS_CONTENT = '''
class Model:
    x = 2
'''


UTIL_ACCESSOR_MODEL_FILE_CONTENT = '''
from model.util import X
from model.submodule.subfile import Y

class Model:
    x = X
    y = Y
'''

UTIL_FILE_CONTENT = '''
X = 1
'''

NEW_UTIL_FILE_CONTENT = '''
X = 2
'''

SUBMODULE_FILE_CONTENT = '''
Y = 3
'''

NEW_SUBMODULE_FILE_CONTENT = '''
Y = 4
'''


def test_model_module_finder():
    with tempfile.TemporaryDirectory() as temp_dir:
        _write_model_module_files(temp_dir, ORIG_MODEL_CLASS_CONTENT)
        with model_class_module_loaded(temp_dir, 'model.model') as model_module:
            model_class = getattr(model_module, 'Model')
            assert model_class.x == 1


def test_model_module_finder_reload():
    with tempfile.TemporaryDirectory() as temp_dir:
        _write_model_module_files(temp_dir, ORIG_MODEL_CLASS_CONTENT)
        with model_class_module_loaded(temp_dir, 'model.model') as model_module:
            model_class = getattr(model_module, 'Model')
            assert model_class.x == 1
    with tempfile.TemporaryDirectory() as temp_dir:
        _write_model_module_files(temp_dir, NEW_MODEL_CLASS_CONTENT)
        with model_class_module_loaded(temp_dir, 'model.model') as model_module:
            model_class = getattr(model_module, 'Model')
            assert model_class.x == 2


def test_model_module_finder_reload_non_model_file_updated():
    with tempfile.TemporaryDirectory() as temp_dir:
        _write_model_module_files(
            temp_dir,
            UTIL_ACCESSOR_MODEL_FILE_CONTENT,
            UTIL_FILE_CONTENT,
            SUBMODULE_FILE_CONTENT,
        )
        with model_class_module_loaded(temp_dir, 'model.model') as model_module:
            model_class = getattr(model_module, 'Model')
            assert model_class.x == 1
            assert model_class.y == 3
    with tempfile.TemporaryDirectory() as temp_dir:
        _write_model_module_files(
            temp_dir,
            UTIL_ACCESSOR_MODEL_FILE_CONTENT,
            NEW_UTIL_FILE_CONTENT,
            NEW_SUBMODULE_FILE_CONTENT,
        )
        with model_class_module_loaded(temp_dir, 'model.model') as model_module:
            model_class = getattr(model_module, 'Model')
            assert model_class.x == 2
            assert model_class.y == 4


def _write_model_module_files(
    scaffold_dir: str,
    model_file_content: str,
    util_file_content: str = None,
    submodule_file_content: str = None,
):
    model_dir_path = Path(scaffold_dir) / 'model'
    model_dir_path.mkdir()
    with (model_dir_path / 'model.py').open('w') as f:
        f.write(model_file_content)
    if util_file_content is not None:
        with (model_dir_path / 'util.py').open('w') as f:
            f.write(util_file_content)

    if submodule_file_content is not None:
        submodule_path = model_dir_path / 'submodule'
        submodule_path.mkdir()
        with (submodule_path / 'subfile.py').open('w') as f:
            f.write(submodule_file_content)
