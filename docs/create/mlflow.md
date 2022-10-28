# Create a Truss of an MLFlow model

[MLFlow](https://mlflow.org/) is a supported framework on Truss. To package an MLFlow model, follow the steps below or run [this Google Colab notebook](https://colab.research.google.com/github/basetenlabs/truss/blob/main/docs/notebooks/mlflow_example.ipynb).

### Install packages

If you're using a Jupyter notebook, add a line to install the `mlflow` and `truss` packages. Otherwise, ensure the packages are installed in your Python environment.

```python
!pip install --upgrade pip
!pip install --upgrade mlflow truss
```

{% hint style="warning" %}
Truss officially supports `mlflow` version 1.30.0 or higher. Especially if you're using an online notebook environment like Google Colab or a bundle of packages like Anaconda, ensure that the version you are using is supported. If it's not, use the `--upgrade` flag and pip will install the most recent version.
{% endhint %}

### Create an in-memory model

This is the part you want to replace with your own code. We are creating a super simple logistic regression model, but you can package any MLFlow model as a Truss.

```python
import mlflow
from sklearn.linear_model import LogisticRegression
import numpy as np

with mlflow.start_run():
    X = np.array([-2, -1, 0, 1, 2, 1]).reshape(-1, 1)
    y = np.array([0, 0, 1, 1, 1, 0])
    lr = LogisticRegression()
    lr.fit(X, y)
    model_info = mlflow.sklearn.log_model(sk_model=lr, artifact_path="model")
    MODEL_URI = model_info.model_uri
```

### Create a Truss

Truss uses MLFlow's [pyfunc](https://www.mlflow.org/docs/latest/python_api/mlflow.pyfunc.html) module in the packaging process. Once you have loaded the model, use the `mk_truss` command to package your model into a Truss.

```python
import os
import truss

model = mlflow.pyfunc.load_model(MODEL_URI)
tr = truss.mk_truss(model, target_directory="./mlflow_truss_from_pyfunc")
```

Check the target directory to see your new Truss!

### Serve the model

To get a prediction from the Truss, try running:

```python
data = np.array([-4, 1, 0, 10, -2, 1]).reshape(-1, 1)
tr.docker_predict(data)
```

For more on running the Truss locally, see [local development](../develop/localhost.md).
