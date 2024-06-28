import contextlib
import importlib
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest
import requests
import yaml

from truss.build import init
from truss.contexts.image_builder.serving_image_builder import (
    ServingImageBuilderContext,
)
from truss.contexts.local_loader.docker_build_emulator import DockerBuildEmulator
from truss.truss_config import DEFAULT_BUNDLED_PACKAGES_DIR
from truss.truss_handle import TrussHandle
from truss.types import Example

CUSTOM_MODEL_CODE = """
class Model:
    def __init__(*args, **kwargs):
        pass

    def load(self):
        pass

    def predict(self, model_input):
        return [1 for i in model_input]
"""

CUSTOM_MODEL_USING_EXTERNAL_PACKAGE_CODE = """
import top_module
import subdir.sub_module
import top_module2
class Model:
    def predict(self, model_input):
        return [1 for i in model_input]
"""

CUSTOM_MODEL_CODE_WITH_PRE_AND_POST_PROCESS = """
class Model:
    def __init__(*args, **kwargs):
        pass

    def load(*args, **kwargs):
        pass

    def preprocess(self, model_input):
        # Adds 1 to all
        return [value + 1 for value in model_input]

    def predict(self, model_input):
        # Returns inputs as predictions
        return {
            'predictions': model_input,
        }

    def postprocess(self, model_output):
        # Adds 2 to all
        return {
            'predictions': [value + 2 for value in model_output['predictions']],
        }
"""

CUSTOM_MODEL_CODE_USING_BUNDLED_PACKAGE = """
from test_package import test

class Model:
    def predict(self, request):
        # Returns inputs as predictions
        return {
            'predictions': [test.X],
        }

"""

CUSTOM_MODEL_CODE_FOR_GPU_TESTING = """
import subprocess

class Model:
    def predict(self, request):
        process = subprocess.run(['nvcc','--version'], check=True, stdout=subprocess.PIPE, universal_newlines=True)
        cuda_version = process.stdout.split('\\n')[-2].split()[-1].split('/')[0].split('_')[-1]
        return {
            'predictions': [{'cuda_version': cuda_version}],
        }
"""

CUSTOM_MODEL_CODE_FOR_SECRETS_TESTING = """
import subprocess

class Model:
    def __init__(self, secrets):
        self._secrets = secrets

    def predict(self, request):
        # Expects instance to be secret name and returns secret value as prediction
        secret_name = request['instances'][0]
        return {
            'predictions': [self._secrets[secret_name]],
        }
"""


# Doesn't implement load
NO_LOAD_CUSTOM_MODEL_CODE = """
class Model:
    def preprocess(self, request):
        return request

    def postprocess(self, request):
        return request

    def predict(self, request):
        return {
            'predictions': [1]
        }
"""


# Doesn't implement predict
NO_PREDICT_CUSTOM_MODEL_CODE = """
class MyModel:
    def load(self):
        pass
"""

LONG_LOAD_MODEL_CODE = """
import time
class Model:
     def load(*args, **kwargs):
        time.sleep(10)
        pass

     def predict(self, model_input):
        return {
            'predictions': model_input,
        }
"""

# Doesn't implement preprocess
NO_PREPROCESS_CUSTOM_MODEL_CODE = """
class Model:
     def load(*args, **kwargs):
        pass

     def postprocess(self, request):
        # Adds 1 to all
        return {
            'predictions': [value + 1 for value in request['predictions']],
        }

     def predict(self, model_input):
        return {
            'predictions': model_input,
        }
"""


# Doesn't implement postprocess
NO_POSTPROCESS_CUSTOM_MODEL_CODE = """
class Model:
     def load(*args, **kwargs):
        pass

     def preprocess(self, model_input):
        # Adds 1 to all
        return [value + 1 for value in model_input]

     def predict(self, model_input):
        return {
            'predictions': model_input,
        }
"""

# Implements no params for init
NO_PARAMS_INIT_CUSTOM_MODEL_CODE = """
class Model:
     def __init__(self):
        pass

     def preprocess(self, request):
        return request

     def postporcess(self, request):
        return request

     def predict(self, model_input):
        return {
            'predictions': model_input,
        }
"""


EXTERNAL_DATA_ACCESS = """
class Model:
    def __init__(self, data_dir):
        self._data_dir = data_dir
        pass

    def predict(self, model_input):
        with (self._data_dir / 'test.txt').open() as file:
            return file.read()
"""


@pytest.fixture
def pytorch_model_init_args():
    return {"arg1": 1, "arg2": 2, "kwarg1": 3, "kwarg2": 4}


@pytest.fixture
def custom_model_truss_dir(tmp_path) -> Path:
    yield _custom_model_from_code(
        tmp_path,
        "custom_truss",
        CUSTOM_MODEL_CODE,
    )


@pytest.fixture
def no_preprocess_custom_model(tmp_path):
    yield _custom_model_from_code(
        tmp_path,
        "my_no_preprocess_model",
        NO_PREPROCESS_CUSTOM_MODEL_CODE,
    )


@pytest.fixture
def long_load_model(tmp_path):
    yield _custom_model_from_code(
        tmp_path,
        "long_load_model",
        LONG_LOAD_MODEL_CODE,
    )


@pytest.fixture
def custom_model_control(tmp_path):
    yield _custom_model_from_code(
        tmp_path,
        "control_truss",
        CUSTOM_MODEL_CODE,
        handle_ops=lambda handle: handle.live_reload(),
    )


@pytest.fixture
def custom_model_external_data_access_tuple_fixture(tmp_path: Path):
    content = "test"
    filename = "test.txt"
    (tmp_path / filename).write_text(content)
    port = 9089
    proc = subprocess.Popen(
        ["python", "-m", "http.server", str(port), "--bind", "*"],
        cwd=tmp_path,
    )
    try:
        url = f"http://localhost:{port}/{filename}"
        # Add arbitrary get params to get that they don't cause issues, the
        # server above ignores them.
        # url_with_get_params = f"{url}?foo=bar&baz=bla"
        url_with_get_params = f"{url}?foo=bar&baz=bla"
        yield (
            _custom_model_from_code(
                tmp_path,
                "external_data_access",
                EXTERNAL_DATA_ACCESS,
                handle_ops=lambda handle: handle.add_external_data_item(
                    url=url_with_get_params, local_data_path="test.txt"
                ),
            ),
            content,
        )
    finally:
        proc.kill()


@pytest.fixture
def custom_model_external_data_access_tuple_fixture_gpu(tmp_path: Path):
    content = "test"
    filename = "test.txt"
    (tmp_path / filename).write_text(content)
    port = 9089
    proc = subprocess.Popen(
        ["python", "-m", "http.server", str(port), "--bind", "*"],
        cwd=tmp_path,
    )
    try:
        url = f"http://localhost:{port}/{filename}"
        # Add arbitrary get params to get that they don't cause issues, the
        # server above ignores them.
        # url_with_get_params = f"{url}?foo=bar&baz=bla"
        url_with_get_params = f"{url}?foo=bar&baz=bla"

        def modify_handle(h):
            h.add_external_data_item(
                url=url_with_get_params, local_data_path="test.txt"
            )
            h.enable_gpu()

        yield (
            _custom_model_from_code(
                tmp_path,
                "external_data_access",
                EXTERNAL_DATA_ACCESS,
                handle_ops=modify_handle,
            ),
            content,
        )
    finally:
        proc.kill()


@pytest.fixture
def custom_model_with_external_package(tmp_path: Path):
    ext_pkg_path = tmp_path / "ext_pkg"
    ext_pkg_path.mkdir()
    (ext_pkg_path / "subdir").mkdir()
    (ext_pkg_path / "subdir" / "sub_module.py").touch()
    (ext_pkg_path / "top_module.py").touch()
    ext_pkg_path2 = tmp_path / "ext_pkg2"
    ext_pkg_path2.mkdir()
    (ext_pkg_path2 / "top_module2.py").touch()

    def add_packages(handle):
        # Use absolute path for this
        handle.add_external_package(str(ext_pkg_path.resolve()))
        # Use relative path for this
        handle.add_external_package("../ext_pkg2")

    yield _custom_model_from_code(
        tmp_path,
        "control_truss",
        CUSTOM_MODEL_USING_EXTERNAL_PACKAGE_CODE,
        handle_ops=add_packages,
    )


@pytest.fixture
def no_postprocess_custom_model(tmp_path):
    yield _custom_model_from_code(
        tmp_path,
        "my_no_postprocess_model",
        NO_POSTPROCESS_CUSTOM_MODEL_CODE,
    )


@pytest.fixture
def no_load_custom_model(tmp_path):
    yield _custom_model_from_code(
        tmp_path,
        "my_no_load_model",
        NO_LOAD_CUSTOM_MODEL_CODE,
    )


@pytest.fixture
def no_params_init_custom_model(tmp_path):
    yield _custom_model_from_code(
        tmp_path,
        "my_no_params_init_load_model",
        NO_PARAMS_INIT_CUSTOM_MODEL_CODE,
    )


@pytest.fixture
def custom_model_trt_llm(tmp_path):
    def modify_handle(h: TrussHandle):
        with _modify_yaml(h.spec.config_path) as content:
            h.enable_gpu()
            content["trt_llm"] = {
                "build": {
                    "base_model": "llama",
                    "max_input_len": 1024,
                    "max_output_len": 1024,
                    "max_batch_size": 512,
                    "max_beam_width": 1,
                    "checkpoint_repository": {
                        "source": "HF",
                        "repo": "meta/llama4-500B",
                    },
                }
            }
            content["resources"]["accelerator"] = "H100:1"

    yield _custom_model_from_code(
        tmp_path,
        "my_trt_llm_model",
        CUSTOM_MODEL_CODE,
        handle_ops=modify_handle,
    )


@pytest.fixture
def useless_file(tmp_path):
    f = tmp_path / "useless.py"
    f.write_text("")
    sys.path.append(str(tmp_path))
    return f


@contextlib.contextmanager
def temp_dir(directory):
    """A context to allow user to drop into the temporary
    directory created by the tmp_path fixture"""
    current_dir = os.getcwd()
    os.chdir(directory)
    try:
        yield
    finally:
        os.chdir(current_dir)


@pytest.fixture
def custom_model_truss_dir_with_pre_and_post_no_example(tmp_path):
    dir_path = tmp_path / "custom_truss_with_pre_post_no_example"
    handle = init(str(dir_path))
    handle.spec.model_class_filepath.write_text(
        CUSTOM_MODEL_CODE_WITH_PRE_AND_POST_PROCESS
    )
    yield dir_path


@pytest.fixture
def custom_model_truss_dir_with_hidden_files(tmp_path):
    truss_dir_path: Path = tmp_path / "custom_model_truss_dir_with_hidden_files"
    _ = init(str(truss_dir_path))
    (truss_dir_path / "__pycache__").mkdir(parents=True, exist_ok=True)
    (truss_dir_path / ".git").mkdir(parents=True, exist_ok=True)
    (truss_dir_path / "__pycache__" / "test.cpython-311.pyc").touch()
    (truss_dir_path / ".DS_Store").touch()
    (truss_dir_path / ".git" / ".test_file").touch()
    (truss_dir_path / "data" / "test_file").write_text("123456789")
    yield truss_dir_path


@pytest.fixture
def custom_model_truss_dir_with_truss_ignore(tmp_path):
    truss_dir_path: Path = tmp_path / "custom_model_truss_dir_with_truss_ignore"
    _ = init(str(truss_dir_path))
    (truss_dir_path / "random_folder_1").mkdir(parents=True, exist_ok=True)
    (truss_dir_path / "random_folder_2").mkdir(parents=True, exist_ok=True)
    (truss_dir_path / "random_file_1.txt").touch()
    (truss_dir_path / "random_folder_1" / "random_file_2.txt").touch()
    (truss_dir_path / "random_folder_2" / "random_file_3.txt").touch()

    (truss_dir_path / ".truss_ignore").write_text(
        """
        random_folder_1
        random_file_1.txt
        """
    )

    yield truss_dir_path


@pytest.fixture
def custom_model_truss_dir_with_pre_and_post(tmp_path):
    dir_path = tmp_path / "custom_truss_with_pre_post"
    handle = init(str(dir_path))
    handle.spec.model_class_filepath.write_text(
        CUSTOM_MODEL_CODE_WITH_PRE_AND_POST_PROCESS
    )
    handle.update_examples([Example("example1", {"inputs": [[0]]})])
    yield dir_path


@pytest.fixture
def custom_model_truss_dir_with_bundled_packages(tmp_path):
    truss_dir_path: Path = tmp_path / "custom_model_truss_dir_with_bundled_packages"
    handle = init(str(truss_dir_path))
    handle.spec.model_class_filepath.write_text(CUSTOM_MODEL_CODE_USING_BUNDLED_PACKAGE)
    packages_path = truss_dir_path / DEFAULT_BUNDLED_PACKAGES_DIR / "test_package"
    packages_path.mkdir(parents=True)
    with (packages_path / "test.py").open("w") as file:
        file.write("""X = 1""")
    yield truss_dir_path


@pytest.fixture
def custom_model_truss_dir_with_pre_and_post_str_example(tmp_path):
    dir_path = tmp_path / "custom_truss_with_pre_post_str_example"
    handle = init(str(dir_path))
    handle.spec.model_class_filepath.write_text(
        CUSTOM_MODEL_CODE_WITH_PRE_AND_POST_PROCESS
    )
    handle.update_examples(
        [
            Example(
                "example1",
                {
                    "inputs": [
                        {
                            "image_url": "https://github.com/pytorch/hub/raw/master/images/dog.jpg"
                        },
                    ],
                },
            )
        ]
    )
    yield dir_path


@pytest.fixture
def custom_model_truss_dir_with_pre_and_post_description(tmp_path):
    dir_path = tmp_path / "custom_truss_with_pre_post"
    handle = init(str(dir_path))
    handle.spec.model_class_filepath.write_text(
        CUSTOM_MODEL_CODE_WITH_PRE_AND_POST_PROCESS
    )
    handle.update_description("This model adds 3 to all inputs")
    yield dir_path


@pytest.fixture
def custom_model_truss_dir_for_gpu(tmp_path):
    dir_path = tmp_path / "custom_truss"
    handle = init(str(dir_path))
    handle.enable_gpu()
    handle.spec.model_class_filepath.write_text(CUSTOM_MODEL_CODE_FOR_GPU_TESTING)
    yield dir_path


@pytest.fixture
def custom_model_truss_dir_for_secrets(tmp_path):
    dir_path = tmp_path / "custom_truss"
    handle = init(str(dir_path))
    handle.add_secret("secret_name", "default_secret_value")
    handle.spec.model_class_filepath.write_text(CUSTOM_MODEL_CODE_FOR_SECRETS_TESTING)
    yield dir_path


@pytest.fixture
def truss_container_fs(tmp_path):
    ROOT = Path(__file__).parent.parent.parent.resolve()
    return _build_truss_fs(ROOT / "truss" / "test_data" / "test_truss", tmp_path)


@pytest.fixture
def truss_control_container_fs(tmp_path):
    ROOT = Path(__file__).parent.parent.parent.resolve()
    test_truss_dir = ROOT / "truss" / "test_data" / "test_truss"
    control_truss_dir = tmp_path / "control_truss"
    shutil.copytree(str(test_truss_dir), str(control_truss_dir))
    with _modify_yaml(control_truss_dir / "config.yaml") as content:
        content["live_reload"] = True
    return _build_truss_fs(control_truss_dir, tmp_path)


@pytest.fixture
def patch_ping_test_server():
    port = "5001"
    proc = subprocess.Popen(
        [
            "poetry",
            "run",
            "flask",
            "--app",
            "app",
            "run",
            "-p",
            port,
            "--host",
            "0.0.0.0",
        ],
        cwd=str(Path(__file__).parent.parent / "test_data" / "patch_ping_test_server"),
    )
    base_url = f"http://127.0.0.1:{port}"
    retry_secs = 10
    sleep_between_retries = 1
    for _ in range(int(retry_secs / sleep_between_retries)):
        time.sleep(sleep_between_retries)
        try:
            resp = requests.get(f"{base_url}/health")
        except requests.exceptions.ConnectionError:
            continue
        if resp.status_code == 200:
            break

    yield port
    proc.terminate()


def _pytorch_model_from_content(
    path: Path,
    content: str,
    model_module_name: str = "my_model",
    model_class_name: str = "MyModel",
    model_filename: str = "my_model.py",
):
    f = path / model_filename
    f.write_text(content)

    sys.path.append(str(path))
    model_class = getattr(importlib.import_module(model_module_name), model_class_name)
    return model_class(), f


def _custom_model_from_code(
    where_dir: Path,
    truss_name: str,
    model_code: str,
    handle_ops: callable = None,
) -> Path:
    dir_path = where_dir / truss_name
    handle = init(str(dir_path))
    if handle_ops is not None:
        handle_ops(handle)
    handle.spec.model_class_filepath.write_text(model_code)
    return dir_path


@pytest.fixture
def custom_model_data_dir(tmp_path: Path):
    data_file = tmp_path / "foo.bar"
    data_file.touch()

    def add_data(handle):
        handle.add_data(str(data_file.resolve()))

    yield _custom_model_from_code(
        tmp_path,
        "data_dir_truss",
        CUSTOM_MODEL_CODE,
        handle_ops=add_data,
    )


class Helpers:
    @staticmethod
    @contextlib.contextmanager
    def file_content(file_path: Path, content: str):
        orig_content = file_path.read_text()
        try:
            file_path.write_text(content)
            yield
        finally:
            file_path.write_text(orig_content)

    @staticmethod
    @contextlib.contextmanager
    def sys_path(path: Path):
        try:
            sys.path.append(str(path))
            yield
        finally:
            sys.path.pop()

    @staticmethod
    @contextlib.contextmanager
    def env_var(var: str, value: str):
        orig_environ = os.environ.copy()
        try:
            os.environ[var] = value
            yield
        finally:
            os.environ.clear()
            os.environ.update(orig_environ)


@pytest.fixture
def helpers():
    return Helpers()


def _build_truss_fs(truss_dir: Path, tmp_path: Path) -> Path:
    truss_fs = tmp_path / "truss_fs"
    truss_fs.mkdir()
    truss_build_dir = tmp_path / "truss_fs_build"
    truss_build_dir.mkdir()
    image_builder = ServingImageBuilderContext.run(truss_dir)
    image_builder.prepare_image_build_dir(truss_build_dir)
    dockerfile_path = truss_build_dir / "Dockerfile"

    docker_build_emulator = DockerBuildEmulator(dockerfile_path, truss_build_dir)
    docker_build_emulator.run(truss_fs)
    return truss_fs


@contextlib.contextmanager
def _modify_yaml(yaml_path: Path):
    with yaml_path.open() as yaml_file:
        content = yaml.safe_load(yaml_file)
    yield content
    with yaml_path.open("w") as yaml_file:
        yaml.dump(content, yaml_file)
