# Create a Truss of a scikit-learn model

[scikit-learn](https://scikit-learn.org/stable/) is a supported framework on Truss. To package a scikit-learn model, follow the steps below or run [this Google Colab notebook](https://colab.research.google.com/github/basetenlabs/truss/blob/main/docs/notebooks/sklearn_example.ipynb).

### Install packages

If you're using a Jupyter notebook, add a line to install the `sklearn` and `truss` packages. Otherwise, ensure the packages are installed in your Python environment.

```python
!pip install --upgrade sklearn truss
```
{% hint style="warning" %}
Truss officially supports `scikit-learn` version 1.0.2 or higher. Especially if you're using an online notebook environment like Google Colab or a bundle of packages like Anaconda, ensure that the version you are using is supported. If it's not, use the `--upgrade` flag and pip will install the most recent version.
{% endhint %}

### Create an in-memory model

This is the part you want to replace with your own code. Using scikit-learn, build a machine learning model and keep it in-memory.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

iris = load_iris()
data_x = iris['data']
data_y = iris['target']
model = RandomForestClassifier()
model.fit(data_x, data_y)
```

### Create a Truss

Use the `create` command to package your model into a Truss.

```python
from truss import create

tr = create(model, target_directory="sklearn_truss")
```

Check the target directory to see your new Truss!

### Serve the model

To get a prediction from the Truss, try running:

```python
tr.docker_predict({"inputs": [[0, 0, 0, 0]]})
```

For more on running the Truss locally, see [local development](../develop/localhost.md).
