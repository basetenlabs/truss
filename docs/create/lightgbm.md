# Create a Truss of a LightGBM model

[LightGBM](https://lightgbm.readthedocs.io/en/v3.3.2/) is a supported framework on Truss. To package a LightGBM model, follow the steps below. 

### Install packages

If you're using a Jupyter notebook, add a line to install the `lightgbm` and `truss` packages. Otherwise, ensure the packages are installed in your Python environment.

```python
!pip install lightgbm truss
```

### Create an in-memory model

This is the part you want to replace with your own code. Using LightGBM, build a machine learning model and keep it in-memory.

```python
import lightgbm as lgb
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

def create_data():
    X, y = make_classification(n_samples=100,
                           n_informative=5,
                           n_classes=2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    train = lgb.Dataset(X_train, y_train)
    test = lgb.Dataset(X_test, y_test)
    return train, test

tparams = {
        'boosting_type': 'gbdt',
        'objective': 'softmax',
        'metric': 'multi_logloss',
        'num_leaves': 31,
        'num_classes': 2,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0
}
model = lgb.train(params=params, train_set=train, valid_sets=test)
```

### Create a Truss

Use the `mk_truss` command to package your model into a Truss. 

```python
from truss import mk_truss

mk_truss(model, target_directory="lightgbm_truss")
```

Check the target directory to see your new Truss!
