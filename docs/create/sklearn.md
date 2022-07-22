# Create a Truss of a scikit-learn model

[scikit-learn](https://scikit-learn.org/stable/) is a supported framework on Truss. To package a scikit-learn model, follow the steps below or run [this colab notebook](https://colab.research.google.com/github/basetenlabs/truss/blob/main/docs/notebooks/sklearn_example.ipynb).

### Install packages

If you're using a Jupyter notebook, add a line to install the `sklearn` and `truss` packages. Otherwise, ensure the packages are installed in your Python environment.

```python
!pip install sklearn truss
```

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

Use the `mk_truss` command to package your model into a Truss.

```python
from truss import mk_truss

tr = mk_truss(model, target_directory="sklearn_truss")
```

Check the target directory to see your new Truss!

### Serve the model

To get a prediction from the Truss, try running:

```python
tr.docker_predict({"inputs": [[0, 0, 0, 0]]})
```

For more on running the Truss locally, see [local development](../develop/localhost.md).