# Extending a Truss

As a motivating example we will look at the case of a hybrid model where we use Keras for modeling with Scikit-learn for preprocessing. In this example we will train a model to determine the MPG of an automobile using the [Auto MPG dataset](http://archive.ics.uci.edu/ml/datasets/Auto+MPG).

The model will be a simple feed forward network in Keras with a standard scaler from Scikit-learn.

```python
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing

from tensorflow.keras import layers

url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'

feature_names =[
    'MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model Year',
]

column_names = feature_names + ['Origin']


raw_dataset = pd.read_csv(url, names=column_names,
                          na_values='?', comment='\t',
                          sep=' ', skipinitialspace=True)

dataset = raw_dataset.copy()
dataset = dataset.dropna()

train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)


train_features = train_dataset.copy().loc[:, feature_names]
test_features = test_dataset.copy().loc[:, feature_names]

train_labels = train_features.pop('MPG')
test_labels = test_features.pop('MPG')

scaler = preprocessing.StandardScaler().fit(train_features)

tf_model = tf.keras.Sequential([
    layers.Dense(units=8),
    layers.Dense(units=1)
])
tf_model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error'
)

history = tf_model.fit(
    scaler.transform(train_features),
    train_labels,
    epochs=100,
    # suppress logging
    verbose=0,
    # Calculate validation results on 20% of the training data
    validation_split = 0.2
)
```

With our trained model; we can create a scaffold for it, here we are doing so at the `test_keras/` directory.

```python
from baseten_scaffolding.scaffold.build import scaffold
scaffold = scaffold(tf_model, target_directory='test_keras')
```

But if we were to serve the model like this, the responses would be garbage-out because we haven't scaled the inputs to feed into the model. To do so we are going to make some edits in the scaffold.

First, we will save a binary of the scaler object we created in the model training phase.

```python
import joblib
joblib.dump(scaler, 'scaler.joblib')
```

Afterwards we will place the `scaler.joblib` file into the `test_keras/src/data/` directory in the scaffold.

```
mv scaler.joblib test_keras/src/data/
```

By default the inference of a Keras model will produce a requirements file that only has `tensorflow` as a requirement. So we will add the following lines to our `requirements.txt` inside of `test_keras/`. Ours were inferred from a `pip list --format=freeze` and might be different than your system.

```
# any previous entries
...
joblib==1.0.1
scikit-learn==0.24.2
```

Then we will make some edits to the scaffold's code to enable it to use the scaler. For some background on the structure of the scaffold check out the [discussion](../discussion/scaffold-structure.md) in our documentation. The file `inference_model.py` is a wrapper around the model you used in construction earlier (`tf_model`) . We will make the following changes to the functions `preprocess` and `load` inside of `test_keras/src/server/inference_model.py` to successfully inject the preprocessing transformations into your model server.

```python
import os
import joblib
...
def load(self):
    self._scaler = joblib.load(os.path.join('data', 'scaler.joblib'))
    ...

def preprocess(self, request: Dict) -> Dict:
    inputs = request['inputs']
    scaled_inputs = self._scaler.transform(np.array(inputs))
    ...
    return {'inputs': scaled_inputs}
```

The `preprocess` method now coerces our input data into a NumPy array and then transforms it according to the scaler we've injected into the scaffold.&#x20;

To test the scaffold locally we can build the docker container, run it, and POST a prediction to it.

We build with a target (`scaffold.docker_build_string` will produce something like this)

```
docker build  -f test_keras/keras-server.Dockerfile test_keras -t test_keras
```

We run with the container's port forwarded

```
docker run --rm  -p 8080:8080 -t test_keras
```

After verifying that the server runs fine, we can test with a cURL command.

```
curl -H 'Content-Type: application/json' \
   -d '{"inputs": [[0,0,0,0,0,0]]}' \
   -X POST http://localhost:8080/v1/models/model:predict
```

##
