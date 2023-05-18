import contextlib
import importlib
import os
import subprocess
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
import requests
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from transformers import AutoModelWithLMHead, AutoTokenizer, pipeline
from truss.build import create, init
from truss.truss_config import DEFAULT_BUNDLED_PACKAGES_DIR
from truss.types import Example
from xgboost import XGBClassifier

PYTORCH_MODEL_FILE_CONTENTS = """
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.main = torch.nn.Sequential(
            torch.nn.Linear(3, 1),
            torch.nn.Flatten(0, 1)
        )

    def forward(self, input):
        return self.main(input)

"""

PYTORCH_MODEL_FILE_WITH_NUMPY_IMPORT_CONTENTS = """
import numpy as np
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.main = torch.nn.Sequential(
            torch.nn.Linear(3, 1),
            torch.nn.Flatten(0, 1)
        )

    def forward(self, input):
        return self.main(input)

"""


PYTORCH_WITH_INIT_ARGS_MODEL_FILE_CONTENTS = """
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, arg1, arg2, kwarg1=1, kwarg2=2):
        super(MyModel, self).__init__()
        self.arg1 = arg1
        self.arg2 = arg2
        self.kwarg1 = kwarg1
        self.kwarg2 = kwarg2

    def forward(self, input):
        return input

"""


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

VARIABLES_TO_ARTIFACTS_TRAIN_CLASS_CODE = """
import json


class Train:
    def __init__(
       self,
       config,
       output_dir,
       variables,
    ):
        self.output_dir = output_dir
        self.variables = variables

    def train(self):
        with (self.output_dir / 'variables.json').open('w') as fp:
            fp.write(json.dumps(self.variables))

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
def pytorch_model(tmp_path):
    return _pytorch_model_from_content(tmp_path, PYTORCH_MODEL_FILE_CONTENTS)


@pytest.fixture
def pytorch_model_with_numpy_import(tmp_path):
    return _pytorch_model_from_content(
        tmp_path, PYTORCH_MODEL_FILE_WITH_NUMPY_IMPORT_CONTENTS
    )


@pytest.fixture
def pytorch_model_init_args():
    return {"arg1": 1, "arg2": 2, "kwarg1": 3, "kwarg2": 4}


@pytest.fixture
def pytorch_model_with_init_args(tmp_path, pytorch_model_init_args):
    f = tmp_path / "my_model_with_init.py"
    f.write_text(PYTORCH_WITH_INIT_ARGS_MODEL_FILE_CONTENTS)

    sys.path.append(str(tmp_path))
    model_class = getattr(importlib.import_module("my_model_with_init"), "MyModel")
    return model_class(**pytorch_model_init_args), f


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
        ["python", "-m", "http.server", str(port), "--bind", "0.0.0.0"],
        cwd=tmp_path,
    )
    try:
        url = f"http://host.docker.internal:{port}/{filename}"
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


@pytest.fixture(scope="session")
def xgboost_pima_model():
    pima_dataset_path = (
        Path(__file__).parent.parent / "test_data" / "pima-indians-diabetes.csv"
    )
    dataset = np.loadtxt(pima_dataset_path, delimiter=",")
    model = XGBClassifier()
    X = dataset[:, 0:8]
    Y = dataset[:, 8]
    model.fit(X, Y)
    return model


@pytest.fixture(scope="session")
def iris_dataset():
    return load_iris()


@pytest.fixture(scope="session")
def sklearn_rfc_model(iris_dataset):
    data_x = iris_dataset["data"]
    data_y = iris_dataset["target"]
    data_x = pd.DataFrame(data_x)
    rfc_model = RandomForestClassifier()
    rfc_model.fit(data_x, data_y)
    return rfc_model


@pytest.fixture(scope="session")
def lgb_pima_model():
    pima_dataset_path = (
        Path(__file__).parent.parent / "test_data" / "pima-indians-diabetes.csv"
    )
    params = {
        "boosting_type": "gbdt",
        "objective": "softmax",
        "metric": "multi_logloss",
        "num_leaves": 31,
        "num_classes": 2,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": 0,
    }
    dataset = pd.read_csv(pima_dataset_path, header=None)
    Y = dataset[8]
    X = dataset.drop(8, axis=1)
    train = lgb.Dataset(X, Y)
    model = lgb.train(params=params, train_set=train)
    return model


@pytest.fixture(scope="session")
def mpg_dataset():
    # url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
    auto_mpg_data = Path(__file__).parent.parent / "test_data" / "auto-mpg.data"
    column_names = [
        "MPG",
        "Cylinders",
        "Displacement",
        "Horsepower",
        "Weight",
        "Acceleration",
        "Model Year",
        "Origin",
    ]

    raw_dataset = pd.read_csv(
        str(auto_mpg_data),
        names=column_names,
        na_values="?",
        comment="\t",
        sep=" ",
        skipinitialspace=True,
    )

    dataset = raw_dataset.copy()
    dataset.isna().sum()
    dataset = dataset.dropna()
    dataset["Origin"] = dataset["Origin"].map({1: "USA", 2: "Europe", 3: "Japan"})
    dataset = pd.get_dummies(dataset, columns=["Origin"], prefix="", prefix_sep="")
    return dataset


@pytest.fixture(scope="session")
def keras_mpg_model(mpg_dataset):
    train_dataset = mpg_dataset.sample(frac=0.8, random_state=0)
    train_features = train_dataset.copy()
    train_labels = train_features.pop("MPG")
    normalizer = preprocessing.Normalization(
        input_shape=[
            9,
        ],
        axis=None,
    )
    normalizer.adapt(np.array(train_features))
    linear_model = tf.keras.Sequential([normalizer, layers.Dense(units=1)])

    linear_model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=0.1), loss="mean_absolute_error"
    )

    linear_model.fit(
        train_features,
        train_labels,
        epochs=5,
        # suppress logging
        verbose=0,
        # Calculate validation results on 20% of the training data
        validation_split=0.2,
    )
    return linear_model


@pytest.fixture(scope="session")
def huggingface_transformer_t5_small_pipeline():
    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    model = AutoModelWithLMHead.from_pretrained("t5-small")
    return pipeline("text2text-generation", model=model, tokenizer=tokenizer)


@pytest.fixture(scope="session")
def huggingface_transformer_t5_small_model():
    model = AutoModelWithLMHead.from_pretrained("t5-small")
    return model


@pytest.fixture(scope="session")
def huggingface_transformer_t5_small_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    return tokenizer


@pytest.fixture
def custom_model_truss_dir_with_pre_and_post_no_example(tmp_path):
    dir_path = tmp_path / "custom_truss_with_pre_post_no_example"
    handle = init(str(dir_path))
    handle.spec.model_class_filepath.write_text(
        CUSTOM_MODEL_CODE_WITH_PRE_AND_POST_PROCESS
    )
    yield dir_path


@pytest.fixture
def huggingface_truss_handle_small_model(
    tmp_path: Path,
    huggingface_transformer_t5_small_pipeline,
):
    dir_path = tmp_path / "huggingface_truss_small_model"
    dir_path.mkdir()
    create(huggingface_transformer_t5_small_pipeline, target_directory=str(dir_path))
    return dir_path


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
def variables_to_artifacts_training_truss(tmp_path):
    dir_path = tmp_path / "training_truss"
    handle = init(str(dir_path), trainable=True)
    handle.spec.train_class_filepath.write_text(VARIABLES_TO_ARTIFACTS_TRAIN_CLASS_CODE)
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
    ROOT = str(Path(__file__).parent.parent.parent.resolve())
    subprocess.run(["truss", "run-image", "truss/test_data/test_truss"], cwd=ROOT)
    truss_fs = tmp_path / "truss_fs"
    truss_fs.mkdir()

    ps_output = subprocess.check_output(
        [
            "docker",
            "ps",
            "--filter",
            "label=truss_dir=truss/test_data/test_truss",
            "--format",
            "'{{.Names}},{{.Image}}'",
        ]
    )
    container_name, image = ps_output.decode("utf-8").strip()[1:-1].split(",")
    subprocess.run(["docker", "cp", f"{container_name}:/app", str(truss_fs / "app")])
    subprocess.run(["docker", "kill", container_name])
    subprocess.run(["docker", "rmi", image, "-f"])
    return truss_fs


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
