import contextlib
import importlib
import os
import shutil
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from transformers import AutoModelWithLMHead, AutoTokenizer, pipeline
from truss.build import init
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

    def predict(self, request):
        return [1 for i in request['inputs']]
"""

CUSTOM_MODEL_CODE_WITH_PRE_AND_POST_PROCESS = """
class Model:
    def __init__(*args, **kwargs):
        pass

    def load(*args, **kwargs):
        pass

    def preprocess(self, request):
        # Adds 1 to all
        return {
            'inputs': [value + 1 for value in request['inputs']]
        }

    def predict(self, request):
        # Returns inputs as predictions
        return {
            'predictions': request['inputs'],
        }

    def postprocess(self, request):
        # Adds 2 to all
        return {
            'predictions': [value + 2 for value in request['predictions']],
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

     def predict(self, request):
        return {
            'predictions': request['inputs'],
        }
"""


# Doesn't implement postprocess
NO_POSTPROCESS_CUSTOM_MODEL_CODE = """
class Model:
     def load(*args, **kwargs):
        pass

     def preprocess(self, request):
        # Adds 1 to all
        return {
            'inputs': [value + 1 for value in request['inputs']],
        }

     def predict(self, request):
        return {
            'predictions': request['inputs'],
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

     def predict(self, request):
        return {
            'predictions': request['inputs'],
        }
"""


@pytest.fixture
def pytorch_model(tmp_path):
    return _pytorch_model_from_content(
        tmp_path,
        PYTORCH_MODEL_FILE_CONTENTS,
        model_module_name="my_model",
        model_class_name="MyModel",
        model_filename="my_model.py",
    )


@pytest.fixture
def pytorch_model_with_numpy_import(tmp_path):
    return _pytorch_model_from_content(
        tmp_path,
        PYTORCH_MODEL_FILE_WITH_NUMPY_IMPORT_CONTENTS,
        model_module_name="my_model",
        model_class_name="MyModel",
        model_filename="my_model.py",
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
    dir_path = tmp_path / "custom_truss"
    handle = init(str(dir_path))
    with handle.spec.model_class_filepath.open("w") as file:
        file.write(CUSTOM_MODEL_CODE)
    return dir_path


@pytest.fixture
def no_preprocess_custom_model(tmp_path):
    dir_path = tmp_path / "my_no_preprocess_model"
    handle = init(str(dir_path))
    with handle.spec.model_class_filepath.open("w") as file:
        file.write(NO_PREPROCESS_CUSTOM_MODEL_CODE)
    yield dir_path


@pytest.fixture
def custom_model_control(tmp_path):
    dir_path = tmp_path / "control_truss"
    handle = init(str(dir_path))
    handle.use_control_plane()
    with handle.spec.model_class_filepath.open("w") as file:
        file.write(CUSTOM_MODEL_CODE)
    yield dir_path


@pytest.fixture
def no_postprocess_custom_model(tmp_path):
    dir_path = tmp_path / "my_no_postprocess_model"
    handle = init(str(dir_path))
    with handle.spec.model_class_filepath.open("w") as file:
        file.write(NO_POSTPROCESS_CUSTOM_MODEL_CODE)
    yield dir_path


@pytest.fixture
def no_load_custom_model(tmp_path):
    dir_path = tmp_path / "my_no_load_model"
    handle = init(str(dir_path))
    with handle.spec.model_class_filepath.open("w") as file:
        file.write(NO_LOAD_CUSTOM_MODEL_CODE)
    yield dir_path


@pytest.fixture
def no_params_init_custom_model(tmp_path):
    dir_path = tmp_path / "my_no_params_init_load_model"
    handle = init(str(dir_path))
    with handle.spec.model_class_filepath.open("w") as file:
        file.write(NO_PARAMS_INIT_CUSTOM_MODEL_CODE)
    yield dir_path


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
    normalizer = preprocessing.Normalization(axis=-1)
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
    with handle.spec.model_class_filepath.open("w") as file:
        file.write(CUSTOM_MODEL_CODE_WITH_PRE_AND_POST_PROCESS)
    yield dir_path


@pytest.fixture
def custom_model_truss_dir_with_pre_and_post(tmp_path):
    dir_path = tmp_path / "custom_truss_with_pre_post"
    handle = init(str(dir_path))
    with handle.spec.model_class_filepath.open("w") as file:
        file.write(CUSTOM_MODEL_CODE_WITH_PRE_AND_POST_PROCESS)
    handle.update_examples([Example("example1", {"inputs": [[0]]})])
    yield dir_path


@pytest.fixture
def custom_model_truss_dir_with_bundled_packages(tmp_path):
    truss_dir_path: Path = tmp_path / "custom_model_truss_dir_with_bundled_packages"
    handle = init(str(truss_dir_path))
    with handle.spec.model_class_filepath.open("w") as file:
        file.write(CUSTOM_MODEL_CODE_USING_BUNDLED_PACKAGE)
    packages_path = truss_dir_path / DEFAULT_BUNDLED_PACKAGES_DIR / "test_package"
    packages_path.mkdir(parents=True)
    with (packages_path / "test.py").open("w") as file:
        file.write("""X = 1""")
    yield truss_dir_path


@pytest.fixture
def custom_model_truss_dir_with_pre_and_post_str_example(tmp_path):
    dir_path = tmp_path / "custom_truss_with_pre_post_str_example"
    handle = init(str(dir_path))
    with handle.spec.model_class_filepath.open("w") as file:
        file.write(CUSTOM_MODEL_CODE_WITH_PRE_AND_POST_PROCESS)
    handle.update_examples(
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
    handle = init(str(dir_path))
    with handle.spec.model_class_filepath.open("w") as file:
        file.write(CUSTOM_MODEL_CODE_WITH_PRE_AND_POST_PROCESS)
    handle.update_description("This model adds 3 to all inputs")
    yield dir_path


@pytest.fixture
def custom_model_truss_dir_for_gpu(tmp_path):
    dir_path = tmp_path / "custom_truss"
    handle = init(str(dir_path))
    handle.enable_gpu()
    with handle.spec.model_class_filepath.open("w") as file:
        file.write(CUSTOM_MODEL_CODE_FOR_GPU_TESTING)
    yield dir_path


@pytest.fixture
def custom_model_truss_dir_for_secrets(tmp_path):
    dir_path = tmp_path / "custom_truss"
    handle = init(str(dir_path))
    handle.add_secret("secret_name", "default_secret_value")
    with handle.spec.model_class_filepath.open("w") as file:
        file.write(CUSTOM_MODEL_CODE_FOR_SECRETS_TESTING)
    yield dir_path


@pytest.fixture
def truss_container_fs(tmp_path):
    truss_fs = tmp_path / "truss_fs"
    truss_fs_test_data_path = (
        Path(__file__).parent.parent / "test_data" / "truss_container_fs"
    )
    shutil.copytree(str(truss_fs_test_data_path), str(truss_fs))
    return truss_fs


def _pytorch_model_from_content(
    path: Path,
    content: str,
    model_module_name: str,
    model_class_name: str,
    model_filename: str,
):
    f = path / model_filename
    f.write_text(content)

    sys.path.append(str(path))
    model_class = getattr(importlib.import_module(model_module_name), model_class_name)
    return model_class(), f
