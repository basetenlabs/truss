# Create a Truss of a XGBoost model

[XGBoost](https://xgboost.readthedocs.io/en/stable/) will soon be a supported framework on Truss, but for now it is still easy to create a XGBoost Truss with a manual export. To package a XGBoost model, follow the steps below or run [this colab notebook]().

### Install packages

If you're using a Jupyter notebook, add a line to install the `xgboost` and `truss` packages. Otherwise, ensure the packages are installed in your Python environment.

```python
!pip install xgboost truss
```

### Create an in-memory model

This is the part you want to replace with your own code. Using a Hugging Face transformer, build a machine learning model and keep it in-memory.

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

### Create a Truss

The `mk_truss` command doesn't yet support XGBoost models (it will soon!). But you can export your XGBoost model to a custom Truss.

Run:

```
truss init xgboost_truss
```

Then:

```python
import os
model.save_model(os.path.join("xgboost_truss", "data", "model", "xgboost.json"))
```

Check the target directory to see your new Truss!