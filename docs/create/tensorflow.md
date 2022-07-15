# Create a Truss of a TensorFlow model

[TensorFlow](https://www.tensorflow.org/) is a supported framework on Truss. To package a TensorFlow model, follow the steps below or run [this colab notebook]().

### Install packages

If you're using a Jupyter notebook, add a line to install the `tensorflow` and `truss` packages. Otherwise, ensure the packages are installed in your Python environment.

```python
!pip install tensorflow truss
# For help installing tensorflow, see https://www.tensorflow.org/install/pip
```

### Create an in-memory model

This is the part you want to replace with your own code. Using TensorFlow, build a machine learning model and keep it in-memory.

```python
import tensorflow as tf

#Creates tensorflow model
model = tf.keras.applications.ResNet50V2(
    include_top=True,
    weights="imagenet",
    classifier_activation="softmax",
)
```

### Create a Truss

Use the `mk_truss` command to package your model into a Truss.

```python
from truss import mk_truss

mk_truss(model, target_directory="tensorflow_truss")
```

Check the target directory to see your new Truss!