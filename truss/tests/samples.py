# flake8: noqa
##############################################################################
import os
from pathlib import Path

PYTORCH_MODEL_CODE = """
import torch
import torch.nn as nn

from utils.myutil import my_util_function

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.main = torch.nn.Sequential(
            torch.nn.Linear(3, 1),
            torch.nn.Flatten(0, 1)
        )

    def forward(self, input):
        print('Using my util function', my_util_function())
        return self.main(input)
"""

UTIL_CODE = """
def my_util_function():
    return 1
"""

PYTORCH_EG_PATH = 'pytorch_eg'
path = Path(PYTORCH_EG_PATH)
path.mkdir(parents=True, exist_ok=True)
path = Path(f'{PYTORCH_EG_PATH}/utils')
path.mkdir(parents=True, exist_ok=True)

with open(f'{PYTORCH_EG_PATH}/my_pytorch_model.py', 'w') as f:
    f.write(PYTORCH_MODEL_CODE)

with open(f'{PYTORCH_EG_PATH}/utils/myutil.py', 'w') as f:
    f.write(UTIL_CODE)

with open(f'{PYTORCH_EG_PATH}/utils/__init__.py', 'w') as f:
    f.write('')

current_dir = os.getcwd()
os.chdir(PYTORCH_EG_PATH)
from my_pytorch_model import MyModel

model = MyModel()
os.chdir(current_dir)

from truss.build import mk_truss

# You can take a whole directory
ms = mk_truss(model, model_files=['pytorch_eg/'], data_files=[], target_directory='test_pytorch')
# You can also take single files or globs
ms2 = mk_truss(model, model_files=['pytorch_eg/utils/*.py', 'pytorch_eg/my_pytorch_model.py'], data_files=[], target_directory='test_pytorch2')

ms.docker_build_string
ms.predict([[0,0,0]])


import random
import string

##############################################################################
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

from truss.build import scaffold

random_suffix = ''.join([random.choice(string.ascii_letters) for _ in range(5)])
rfc = RandomForestClassifier()

iris = load_iris()
feature_names = iris['feature_names']
class_labels = list(iris['target_names'])
data_x = iris['data']
data_y = iris['target']
data_x = pd.DataFrame(data_x, columns=feature_names)
rfc.fit(data_x, data_y)

ms = mk_truss(rfc, model_files=[], data_files=[])
ms.docker_build_string
ms.predict([[0,0,0,0]])


##############################################################################
EMBEDDING_REQUIREMENTS = """
tensorflow-hub==0.10.0
tensorflow==2.5.0
scikit-learn==1.0.2
"""

EMBEDDING_UTIL_CODE = """
import numpy as np


def get_top_k_indices(arr, k=5):
    return arr.argsort(axis=0)[-k::][::-1].tolist()


def create_reference_embeddings():
    random_fake_embeddings = np.random.randn(100,512)/10
    np.save('embeddings.npy', random_fake_embeddings)

"""

EMBEDDING_MODEL_CODE = """
import numpy as np
import tensorflow_hub as hub
from sklearn.metrics.pairwise import cosine_similarity
import pathlib

from utils.embedding_util import get_top_k_indices

ENCODER_URL = 'https://tfhub.dev/google/universal-sentence-encoder/4'
REFERENCE_EMBED = 'embeddings.npy'


class MyEmbeddingModel:
    def __init__(self):
        self.embed = None
        self.reference_embeddings = None

    def load(self):
        self.embed = hub.load(ENCODER_URL)
        self.reference_embeddings = np.load(pathlib.Path('model', REFERENCE_EMBED).as_posix())

    def predict(self, inputs):
        # do the prediction
        embedding = self.embed(inputs)
        # compare the embedding with a reference dataset
        embedding_cosine_similarity = cosine_similarity(self.reference_embeddings, embedding)
        # find the most similar
        top_k_similar = get_top_k_indices(embedding_cosine_similarity)
        # return raw and processed results
        return {
            "embedding": embedding.numpy().tolist(),
            "top_k_similar": top_k_similar,
        }

"""
from pathlib import Path

from truss.build import scaffold_custom

path = Path('test_folder')
path.mkdir(parents=True, exist_ok=True)
path = Path('test_folder/utils')
path.mkdir(parents=True, exist_ok=True)
# Create the model file
with open('test_folder/embedding_model.py', 'w') as f:
    f.write(EMBEDDING_MODEL_CODE)

# Create some utils file
with open('test_folder/utils/embedding_util.py', 'w') as f:
    f.write(EMBEDDING_UTIL_CODE)

with open('test_folder/utils/__init__.py', 'w') as f:
    f.write('')

# Create the requirements file
with open('test_folder/embedding_reqs.txt', 'w') as f:
    f.write(EMBEDDING_REQUIREMENTS)

# Create a fake dataset to include with the deployment
from test_folder.utils.embedding_util import create_reference_embeddings

create_reference_embeddings()

mk_truss = scaffold_custom(
    model_files=['test_folder', 'test_folder/utils/*.py', 'embeddings.npy'],
    target_directory='test_custom',
    requirements_file='test_folder/embedding_reqs.txt',
    model_class='MyEmbeddingModel'
)
mk_truss.docker_build_string
mk_truss.predict(['hello world', 'bar baz'])


##############################################################################
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']

raw_dataset = pd.read_csv(url, names=column_names,
                          na_values='?', comment='\t',
                          sep=' ', skipinitialspace=True)

dataset = raw_dataset.copy()
dataset.isna().sum()

dataset = dataset.dropna()
dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})
dataset = pd.get_dummies(dataset, columns=['Origin'], prefix='', prefix_sep='')

train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

train_dataset.describe().transpose()

train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('MPG')
test_labels = test_features.pop('MPG')

normalizer = preprocessing.Normalization(axis=-1)
normalizer.adapt(np.array(train_features))

linear_model = tf.keras.Sequential([
    normalizer,
    layers.Dense(units=1)
])

linear_model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error'
)

history = linear_model.fit(
    train_features, train_labels,
    epochs=100,
    # suppress logging
    verbose=0,
    # Calculate validation results on 20% of the training data
    validation_split = 0.2
)

linear_model.predict(train_features[:10])

from truss.build import scaffold

scaff = mk_truss(model=linear_model)
scaff.docker_build_string
scaff.predict([[0, 0, 0, 0, 0, 0, 0, 0, 0]])

##############################################################################

from truss.constants import HUGGINGFACE_TRANSFORMER
from truss.definitions.base import build_scaffold_directory
from truss.definitions.huggingface_transformer import \
    HuggingFaceTransformerPipelineScaffold

built_scaffold_dir = build_scaffold_directory(
    HUGGINGFACE_TRANSFORMER
)
mk_truss = HuggingFaceTransformerPipelineScaffold(model_type='text-generation', path_to_scaffold=built_scaffold_dir)
mk_truss.predict([{'text_inputs': 'hello world'}])


##############################################################################
import os
from pathlib import Path

PYTORCH_MODEL_CODE = """
import torch
import torch.nn as nn

class MyModelWithArgs(nn.Module):
    def __init__(self, nlayers, layer_size=64):
        super(MyModelWithArgs, self).__init__()
        self.nlayers = nlayers
        self.layer_size = layer_size
        self.input_layer = nn.Linear(3, self.layer_size)
        self.dense_layers = nn.ModuleList([nn.Linear(self.layer_size, self.layer_size) for _ in range(self.nlayers)])
        self.output_layer = nn.Linear(self.layer_size, 1)

    def forward(self, input):
        x = self.input_layer(input)
        for layer in self.dense_layers:
            x = torch.nn.functional.relu(layer(x))
        x = self.output_layer(x)
        return x

"""

PYTORCH_WITH_ARGS_PATH = 'pytorch_with_args'
MODEL_INIT_ARGS = {'nlayers': 2, 'layer_size': 32}

path = Path(PYTORCH_WITH_ARGS_PATH)
path.mkdir(parents=True, exist_ok=True)

with open(f'{PYTORCH_WITH_ARGS_PATH}/my_pytorch_model.py', 'w') as f:
    f.write(PYTORCH_MODEL_CODE)


current_dir = os.getcwd()
os.chdir(PYTORCH_WITH_ARGS_PATH)
from my_pytorch_model import MyModelWithArgs

model = MyModelWithArgs(**MODEL_INIT_ARGS)
os.chdir(current_dir)

from truss.build import scaffold

ms = mk_truss(model, model_files=[PYTORCH_WITH_ARGS_PATH], data_files=[], target_directory='test_pytorch_with_args', model_init_parameters=MODEL_INIT_ARGS)
ms.predict([[0, 0, 0]])


##############################################################################
# Simple custom with args
import os
from pathlib import Path

CUSTOM_MODEL_CODE = """
class MyCustomModelWithArgs:
    def __init__(self, arg1, arg2, keyword_arg=64):
        self.arg1 = arg1
        self.arg2 = arg2
        self.keyword_arg = keyword_arg
        print(f'arg1: {self.arg1}')
        print(f'arg2: {self.arg2}')
        print(f'keyword_arg: {self.keyword_arg}')

    def load(self):
        print('loading model')
        print(f'arg1: {self.arg1}')
        print(f'arg2: {self.arg2}')
        print(f'keyword_arg: {self.keyword_arg}')

    def predict(self, input):
        return input

"""

CUSTOM_WITH_ARGS_PATH = 'custom_with_args'
MODEL_INIT_ARGS = {'arg1': 2, 'arg2': 32, 'keyword_arg': 64}

path = Path(CUSTOM_WITH_ARGS_PATH)
path.mkdir(parents=True, exist_ok=True)

with open(f'{CUSTOM_WITH_ARGS_PATH}/my_custom_model.py', 'w') as f:
    f.write(CUSTOM_MODEL_CODE)

from truss.build import scaffold_custom

ms = scaffold_custom(
    model_files=[CUSTOM_WITH_ARGS_PATH],
    target_directory='test_custom_with_args',
    model_class='MyCustomModelWithArgs',
    model_init_parameters=MODEL_INIT_ARGS,
)
ms.predict([[0, 0, 0]])


##############################################################################
# Simple custom without args
import os
from pathlib import Path

CUSTOM_MODEL_CODE = """
class MyCustomModel:
    def load(self):
        print('loading model')

    def predict(self, input):
        return input

"""

CUSTOM_WITHOUT_ARGS_PATH = 'custom_without_args'

path = Path(CUSTOM_WITHOUT_ARGS_PATH)
path.mkdir(parents=True, exist_ok=True)

with open(f'{CUSTOM_WITHOUT_ARGS_PATH}/my_custom_model.py', 'w') as f:
    f.write(CUSTOM_MODEL_CODE)

from truss.build import scaffold_custom

ms = scaffold_custom(
    model_files=[CUSTOM_WITHOUT_ARGS_PATH],
    target_directory='test_custom_without_args',
    model_class='MyCustomModel',
)
ms.predict([[0, 0, 0]])
