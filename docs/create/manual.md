# Manually

Creating a Truss manually, from a serialized model, works with any model-building framework, including from-scratch bespoke models.

To get started, initialize the Truss with the following command in the CLI:

```
truss init my_truss
```

### Truss structure

To build a Truss manually, you have to understand the package in much more detail than using it with a supported framework. Fortunately, that's what this doc is for!

To familiarize yourself with the structure of Truss, review the [structure reference](../reference/structure.md). A Truss only has a few files that you need to interact with, and this tutorial is an opinionated guide to working through them.

### Adding the model binary

First, you'll need to add a model binary to your new Truss. On supported frameworks, this is provided automatically by the `mk_truss` command. For a custom Truss, it can come from many sources, such as:

* Pickling your model
* Serializing your model
* Downloading a serialized model from the internet

This file should be put in the folder `data/model/` as, for example, `model.joblib` (replace `joblib` with the appropriate extension for your serialized model).

This model binary must be de-serialized in the model class.

### Building the model

The model file implements the following functions, in order of execution:

* A constructor `__init__` to initiate the class
* A function called `load`, called **only** once, and that call is guaranteed to happen before **any** predictions are run
* A function `preprocess`, called once before **each** prediction
* A function `predict` that actually runs the model to make a prediction
* A function `postprocess`, called once after **each** prediction

Having both a constructor and a load function means you have flexibility on when you download and/or deserialize your model. There are three possibilities here, and we strongly recommend the first one:

1. Load in the load function
2. Load model in the constructor, but it's not a good idea to block constructor
3. Load lazily on first prediction, but this gives your model service a cold start issue

Also, your model gets access to certain values, including the `config.yaml` file for configuration and the `data` folder where you previously put the serialized model.

## Example code

While XGBoost is a supported framework — you can make a Truss from an XGBoost model with `mk_truss` — we'll use the manual method here for demonstration.

If you haven't already, create a Truss by running:

```
truss init my_truss
```

This is the part you want to replace with your own code. Build a machine learning model and keep it in-memory.

```python
import xgboost as xgb
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

def create_data():
    X, y = make_classification(n_samples=100,
                               n_informative=5,
                               n_classes=2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    train = xgb.DMatrix(X_train, y_train)
    test = xgb.DMatrix(X_test, y_test)
    return train, test

train, test = create_data()

params = {
    "learning_rate": 0.01,
    "max_depth": 3
}

# training, we set the early stopping rounds parameter
model = xgb.train(params,
        train, evals=[(train, "train"), (test, "validation")],
        num_boost_round=100, early_stopping_rounds=20)
```

Now, we'll serialize and save the model:

```python
import os
model.save_model(os.path.join("my_truss", "data", "model", "xgboost.json"))
```

Once your model is created, you'll likely need to develop it further, see the next section for everything you need to know about local development!
