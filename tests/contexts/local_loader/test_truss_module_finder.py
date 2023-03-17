import tempfile
from pathlib import Path

from truss.contexts.local_loader.truss_module_loader import truss_module_loaded
from truss.truss_config import DEFAULT_BUNDLED_PACKAGES_DIR

ORIG_MODEL_CLASS_CONTENT = """
class Model:
    x = 1
"""

NEW_MODEL_CLASS_CONTENT = """
class Model:
    x = 2
"""


UTIL_ACCESSOR_MODEL_FILE_CONTENT = """
from model.util import X
from model.submodule.subfile import Y

class Model:
    x = X
    y = Y
"""

UTIL_FILE_CONTENT = """
X = 1
"""

NEW_UTIL_FILE_CONTENT = """
X = 2
"""

SUBMODULE_FILE_CONTENT = """
Y = 3
"""

NEW_SUBMODULE_FILE_CONTENT = """
Y = 4
"""

ORIG_MODEL_CLASS_USING_ADDITIONAL_MODULE_CONTENT = """
from test_module import test
class Model:
    x = test.A
"""

ORIG_ADDITIONAL_MODULE_CONTENT = """
A = 1
"""

NEW_ADDITIONAL_MODULE_CONTENT = """
A = 2
"""


def test_model_module_finder():
    with tempfile.TemporaryDirectory() as temp_dir:
        _write_model_module_files(temp_dir, ORIG_MODEL_CLASS_CONTENT)
        with truss_module_loaded(temp_dir, "model.model") as model_module:
            model_class = getattr(model_module, "Model")
            assert model_class.x == 1


def test_model_module_finder_reload():
    with tempfile.TemporaryDirectory() as temp_dir:
        _write_model_module_files(temp_dir, ORIG_MODEL_CLASS_CONTENT)
        with truss_module_loaded(temp_dir, "model.model") as model_module:
            model_class = getattr(model_module, "Model")
            assert model_class.x == 1
    with tempfile.TemporaryDirectory() as temp_dir:
        _write_model_module_files(temp_dir, NEW_MODEL_CLASS_CONTENT)
        with truss_module_loaded(temp_dir, "model.model") as model_module:
            model_class = getattr(model_module, "Model")
            assert model_class.x == 2


def test_model_module_finder_additional_modules():
    with tempfile.TemporaryDirectory() as temp_dir:
        _write_model_module_files(
            temp_dir,
            ORIG_MODEL_CLASS_USING_ADDITIONAL_MODULE_CONTENT,
            additional_module_file_content=ORIG_ADDITIONAL_MODULE_CONTENT,
        )
        with truss_module_loaded(
            temp_dir, "model.model", DEFAULT_BUNDLED_PACKAGES_DIR
        ) as model_module:
            model_class = getattr(model_module, "Model")
            assert model_class.x == 1


def test_model_module_finder_additional_modules_reload():
    with tempfile.TemporaryDirectory() as temp_dir:
        _write_model_module_files(
            temp_dir,
            ORIG_MODEL_CLASS_USING_ADDITIONAL_MODULE_CONTENT,
            additional_module_file_content=ORIG_ADDITIONAL_MODULE_CONTENT,
        )
        with truss_module_loaded(
            temp_dir, "model.model", DEFAULT_BUNDLED_PACKAGES_DIR
        ) as model_module:
            model_class = getattr(model_module, "Model")
            assert model_class.x == 1

    with tempfile.TemporaryDirectory() as temp_dir:
        _write_model_module_files(
            temp_dir,
            ORIG_MODEL_CLASS_USING_ADDITIONAL_MODULE_CONTENT,
            additional_module_file_content=NEW_ADDITIONAL_MODULE_CONTENT,
        )
        with truss_module_loaded(
            temp_dir, "model.model", DEFAULT_BUNDLED_PACKAGES_DIR
        ) as model_module:
            model_class = getattr(model_module, "Model")
            assert model_class.x == 2


def test_model_module_finder_reload_non_model_file_updated():
    with tempfile.TemporaryDirectory() as temp_dir:
        _write_model_module_files(
            temp_dir,
            UTIL_ACCESSOR_MODEL_FILE_CONTENT,
            UTIL_FILE_CONTENT,
            SUBMODULE_FILE_CONTENT,
        )
        with truss_module_loaded(temp_dir, "model.model") as model_module:
            model_class = getattr(model_module, "Model")
            assert model_class.x == 1
            assert model_class.y == 3
    with tempfile.TemporaryDirectory() as temp_dir:
        _write_model_module_files(
            temp_dir,
            UTIL_ACCESSOR_MODEL_FILE_CONTENT,
            NEW_UTIL_FILE_CONTENT,
            NEW_SUBMODULE_FILE_CONTENT,
        )
        with truss_module_loaded(temp_dir, "model.model") as model_module:
            model_class = getattr(model_module, "Model")
            assert model_class.x == 2
            assert model_class.y == 4


def _write_model_module_files(
    truss_dir: str,
    model_file_content: str,
    util_file_content: str = None,
    submodule_file_content: str = None,
    additional_module_file_content: str = None,
):
    model_dir_path = Path(truss_dir) / "model"
    model_dir_path.mkdir(parents=True)
    with (model_dir_path / "model.py").open("w") as f:
        f.write(model_file_content)
    if util_file_content is not None:
        with (model_dir_path / "util.py").open("w") as f:
            f.write(util_file_content)

    if submodule_file_content is not None:
        submodule_path = model_dir_path / "submodule"
        submodule_path.mkdir()
        with (submodule_path / "subfile.py").open("w") as f:
            f.write(submodule_file_content)

    if additional_module_file_content is not None:
        test_module_path = (
            Path(truss_dir) / DEFAULT_BUNDLED_PACKAGES_DIR / "test_module"
        )
        test_module_path.mkdir(parents=True)
        with (test_module_path / "test.py").open("w") as f:
            f.write(additional_module_file_content)
