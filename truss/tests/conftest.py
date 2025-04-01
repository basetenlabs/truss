import contextlib
import copy
import importlib
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict

import pytest
import requests
import yaml

from truss.base.custom_types import Example
from truss.base.trt_llm_config import TrussSpecDecMode, TrussTRTLLMBatchSchedulerPolicy
from truss.base.truss_config import DEFAULT_BUNDLED_PACKAGES_DIR, Accelerator
from truss.contexts.image_builder.serving_image_builder import (
    ServingImageBuilderContext,
)
from truss.contexts.local_loader.docker_build_emulator import DockerBuildEmulator
from truss.truss_handle.build import init_directory
from truss.truss_handle.truss_handle import TrussHandle

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

CUSTOM_MODEL_TRT_LLM_CODE = """
class Model:
    def __init__(trt_llm, *args, **kwargs):
        pass

    def load(self):
        pass

    def predict(self, model_input):
        return [1 for i in model_input]
"""


@pytest.fixture
def test_data_path() -> Path:
    return Path(__file__).parent.resolve() / "test_data"


@pytest.fixture
def pytorch_model_init_args():
    return {"arg1": 1, "arg2": 2, "kwarg1": 3, "kwarg2": 4}


@pytest.fixture
def custom_model_truss_dir(tmp_path) -> Path:
    yield _custom_model_from_code(tmp_path, "custom_truss", CUSTOM_MODEL_CODE)


@pytest.fixture
def no_preprocess_custom_model(tmp_path):
    yield _custom_model_from_code(
        tmp_path, "my_no_preprocess_model", NO_PREPROCESS_CUSTOM_MODEL_CODE
    )


@pytest.fixture
def long_load_model(tmp_path):
    yield _custom_model_from_code(tmp_path, "long_load_model", LONG_LOAD_MODEL_CODE)


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
        ["python", "-m", "http.server", str(port), "--bind", "*"], cwd=tmp_path
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
        ["python", "-m", "http.server", str(port), "--bind", "*"], cwd=tmp_path
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
        tmp_path, "my_no_postprocess_model", NO_POSTPROCESS_CUSTOM_MODEL_CODE
    )


@pytest.fixture
def no_load_custom_model(tmp_path):
    yield _custom_model_from_code(
        tmp_path, "my_no_load_model", NO_LOAD_CUSTOM_MODEL_CODE
    )


@pytest.fixture
def no_params_init_custom_model(tmp_path):
    yield _custom_model_from_code(
        tmp_path, "my_no_params_init_load_model", NO_PARAMS_INIT_CUSTOM_MODEL_CODE
    )


@pytest.fixture
def custom_model_trt_llm(tmp_path):
    def modify_handle(h: TrussHandle):
        with _modify_yaml(h.spec.config_path) as content:
            content["trt_llm"] = {
                "build": {
                    "base_model": "llama",
                    "max_seq_len": 2048,
                    "max_batch_size": 512,
                    "checkpoint_repository": {
                        "source": "HF",
                        "repo": "meta/llama4-500B",
                    },
                },
                "runtime": {
                    "kv_cache_free_gpu_mem_fraction": 0.9,
                    "kv_cache_host_memory_bytes": 1000,
                    "enabled_chunked_context": True,
                    "batch_scheduler_policy": TrussTRTLLMBatchSchedulerPolicy.GUARANTEED_NO_EVICT.value,
                },
            }
            content["resources"] = {"accelerator": "H100:1"}

    yield _custom_model_from_code(
        tmp_path,
        "my_trt_llm_model",
        CUSTOM_MODEL_TRT_LLM_CODE,
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
def dynamic_config_mount_dir(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "truss.templates.shared.dynamic_config_resolver.DYNAMIC_CONFIG_MOUNT_DIR",
        str(tmp_path),
    )
    yield


@pytest.fixture
def custom_model_truss_dir_with_pre_and_post_no_example(tmp_path):
    dir_path = tmp_path / "custom_truss_with_pre_post_no_example"
    th = TrussHandle(init_directory(dir_path))
    th.spec.model_class_filepath.write_text(CUSTOM_MODEL_CODE_WITH_PRE_AND_POST_PROCESS)
    yield dir_path


@pytest.fixture
def custom_model_truss_dir_with_hidden_files(tmp_path):
    truss_dir_path: Path = tmp_path / "custom_model_truss_dir_with_hidden_files"
    init_directory(truss_dir_path)
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
    init_directory(truss_dir_path)
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
    th = TrussHandle(init_directory(dir_path))
    th.spec.model_class_filepath.write_text(CUSTOM_MODEL_CODE_WITH_PRE_AND_POST_PROCESS)
    th.update_examples([Example("example1", {"inputs": [[0]]})])
    yield dir_path


@pytest.fixture
def custom_model_truss_dir_with_bundled_packages(tmp_path):
    truss_dir_path: Path = tmp_path / "custom_model_truss_dir_with_bundled_packages"
    th = TrussHandle(init_directory(truss_dir_path))
    th.spec.model_class_filepath.write_text(CUSTOM_MODEL_CODE_USING_BUNDLED_PACKAGE)
    packages_path = truss_dir_path / DEFAULT_BUNDLED_PACKAGES_DIR / "test_package"
    packages_path.mkdir(parents=True)
    with (packages_path / "test.py").open("w") as file:
        file.write("""X = 1""")
    yield truss_dir_path


@pytest.fixture
def custom_model_truss_dir_with_pre_and_post_str_example(tmp_path):
    dir_path = tmp_path / "custom_truss_with_pre_post_str_example"
    th = TrussHandle(init_directory(dir_path))
    th.spec.model_class_filepath.write_text(CUSTOM_MODEL_CODE_WITH_PRE_AND_POST_PROCESS)
    th.update_examples(
        [
            Example(
                "example1",
                {
                    "inputs": [
                        {
                            "image_url": "https://github.com/pytorch/hub/raw/master/images/dog.jpg"
                        }
                    ]
                },
            )
        ]
    )
    yield dir_path


@pytest.fixture
def custom_model_truss_dir_with_pre_and_post_description(tmp_path):
    dir_path = tmp_path / "custom_truss_with_pre_post"
    th = TrussHandle(init_directory(dir_path))
    th.spec.model_class_filepath.write_text(CUSTOM_MODEL_CODE_WITH_PRE_AND_POST_PROCESS)
    th.update_description("This model adds 3 to all inputs")
    yield dir_path


@pytest.fixture
def custom_model_truss_dir_for_gpu(tmp_path):
    dir_path = tmp_path / "custom_truss"
    th = TrussHandle(init_directory(dir_path))
    th.spec.model_class_filepath.write_text(CUSTOM_MODEL_CODE_FOR_GPU_TESTING)
    yield dir_path


@pytest.fixture
def custom_model_truss_dir_for_secrets(tmp_path):
    dir_path = tmp_path / "custom_truss"
    th = TrussHandle(init_directory(dir_path))
    th.add_secret("secret_name", "default_secret_value")
    th.spec.model_class_filepath.write_text(CUSTOM_MODEL_CODE_FOR_SECRETS_TESTING)
    yield dir_path


@pytest.fixture
def truss_container_fs(tmp_path, test_data_path):
    return _build_truss_fs(test_data_path / "test_truss", tmp_path)


@pytest.fixture
def trt_llm_truss_container_fs(tmp_path, test_data_path):
    return _build_truss_fs(test_data_path / "test_trt_llm_truss", tmp_path)


@pytest.fixture
def open_ai_container_fs(tmp_path, test_data_path):
    return _build_truss_fs(test_data_path / "test_openai", tmp_path)


@pytest.fixture
def truss_control_container_fs(tmp_path, test_data_path):
    test_truss_dir = test_data_path / "test_truss"
    control_truss_dir = tmp_path / "control_truss"
    shutil.copytree(str(test_truss_dir), str(control_truss_dir))
    with _modify_yaml(control_truss_dir / "config.yaml") as content:
        content["live_reload"] = True
    return _build_truss_fs(control_truss_dir, tmp_path)


@pytest.fixture
def patch_ping_test_server(test_data_path):
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
        cwd=str(test_data_path / "patch_ping_test_server"),
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
    where_dir: Path, truss_name: str, model_code: str, handle_ops: callable = None
) -> Path:
    dir_path = where_dir / truss_name
    th = TrussHandle(init_directory(dir_path))
    if handle_ops is not None:
        handle_ops(th)
    th.spec.model_class_filepath.write_text(model_code)
    return dir_path


@pytest.fixture
def custom_model_data_dir(tmp_path: Path):
    data_file = tmp_path / "foo.bar"
    data_file.touch()

    def add_data(handle):
        handle.add_data(str(data_file.resolve()))

    yield _custom_model_from_code(
        tmp_path, "data_dir_truss", CUSTOM_MODEL_CODE, handle_ops=add_data
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
    def sys_paths(*paths: Path):
        num_paths = len(paths)
        try:
            for path in paths:
                sys.path.append(str(path))
            yield
        finally:
            for _ in range(num_paths):
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
        yaml.safe_dump(content, yaml_file)


@pytest.fixture
def default_config() -> Dict[str, Any]:
    return {
        "python_version": "py39",
        "resources": {
            "accelerator": None,
            "cpu": "1",
            "memory": "2Gi",
            "use_gpu": False,
        },
    }


@pytest.fixture
def trtllm_config(default_config) -> Dict[str, Any]:
    trtllm_config = default_config
    trtllm_config["resources"] = {
        "accelerator": Accelerator.L4.value,
        "cpu": "1",
        "memory": "24Gi",
        "use_gpu": True,
        "node_count": 1,
    }
    trtllm_config["trt_llm"] = {
        "build": {
            "base_model": "llama",
            "max_seq_len": 2048,
            "max_batch_size": 512,
            "checkpoint_repository": {"source": "HF", "repo": "meta/llama4-500B"},
            "gather_all_token_logits": False,
        },
        "runtime": {},
    }
    return trtllm_config


@pytest.fixture
def trtllm_config_encoder(default_config) -> Dict[str, Any]:
    trtllm_config = default_config
    trtllm_config["resources"] = {
        "accelerator": Accelerator.L4.value,
        "cpu": "1",
        "memory": "24Gi",
        "use_gpu": True,
        "node_count": 1,
    }
    trtllm_config["trt_llm"] = {
        "build": {
            "base_model": "encoder",
            "checkpoint_repository": {"source": "HF", "repo": "BAAI/bge-m3"},
        },
        "runtime": {},
    }
    return trtllm_config


@pytest.fixture
def deprecated_trtllm_config(default_config) -> Dict[str, Any]:
    trtllm_config = default_config
    trtllm_config["resources"] = {
        "accelerator": Accelerator.L4.value,
        "cpu": "1",
        "memory": "24Gi",
        "use_gpu": True,
    }
    trtllm_config["trt_llm"] = {
        "build": {
            "base_model": "llama",
            "max_seq_len": 2048,
            "max_batch_size": 512,
            # start deprecated fields
            "kv_cache_free_gpu_mem_fraction": 0.1,
            "enable_chunked_context": True,
            "batch_scheduler_policy": TrussTRTLLMBatchSchedulerPolicy.MAX_UTILIZATION.value,
            "request_default_max_tokens": 10,
            "total_token_limit": 50,
            # end deprecated fields
            "checkpoint_repository": {"source": "HF", "repo": "meta/llama4-500B"},
            "gather_all_token_logits": False,
        }
    }
    return trtllm_config


@pytest.fixture
def deprecated_trtllm_config_with_runtime_existing(default_config) -> Dict[str, Any]:
    trtllm_config = default_config
    trtllm_config["resources"] = {
        "accelerator": Accelerator.L4.value,
        "cpu": "1",
        "memory": "24Gi",
        "use_gpu": True,
    }
    trtllm_config["trt_llm"] = {
        "build": {
            "base_model": "llama",
            "max_seq_len": 2048,
            "max_batch_size": 512,
            # start deprecated fields
            "kv_cache_free_gpu_mem_fraction": 0.1,
            "enable_chunked_context": True,
            "batch_scheduler_policy": TrussTRTLLMBatchSchedulerPolicy.MAX_UTILIZATION.value,
            "request_default_max_tokens": 10,
            "total_token_limit": 50,
            # end deprecated fields
            "checkpoint_repository": {"source": "HF", "repo": "meta/llama4-500B"},
            "gather_all_token_logits": False,
        },
        "runtime": {"total_token_limit": 100},
    }
    return trtllm_config


@pytest.fixture
def trtllm_spec_dec_config_full(trtllm_config) -> Dict[str, Any]:
    spec_dec_config = copy.deepcopy(trtllm_config)
    spec_dec_config["trt_llm"] = {
        "build": {
            "base_model": "llama",
            "max_seq_len": 2048,
            "max_batch_size": 512,
            "checkpoint_repository": {"source": "HF", "repo": "meta/llama4-500B"},
            "plugin_configuration": {
                "paged_kv_cache": True,
                "use_paged_context_fmha": True,
            },
            "speculator": {
                "speculative_decoding_mode": TrussSpecDecMode.DRAFT_EXTERNAL.value,
                "num_draft_tokens": 4,
                "build": {
                    "base_model": "llama",
                    "max_seq_len": 2048,
                    "max_batch_size": 512,
                    "checkpoint_repository": {
                        "source": "HF",
                        "repo": "meta/llama4-500B",
                    },
                },
            },
        }
    }
    return spec_dec_config


@pytest.fixture
def trtllm_spec_dec_config(trtllm_config) -> Dict[str, Any]:
    spec_dec_config = copy.deepcopy(trtllm_config)
    spec_dec_config["trt_llm"] = {
        "build": {
            "base_model": "llama",
            "max_seq_len": 2048,
            "max_batch_size": 512,
            "checkpoint_repository": {"source": "HF", "repo": "meta/llama4-500B"},
            "plugin_configuration": {
                "paged_kv_cache": True,
                "use_paged_context_fmha": True,
            },
            "speculator": {
                "speculative_decoding_mode": TrussSpecDecMode.DRAFT_EXTERNAL.value,
                "num_draft_tokens": 4,
                "checkpoint_repository": {"source": "HF", "repo": "meta/llama4-500B"},
            },
        }
    }
    return spec_dec_config
