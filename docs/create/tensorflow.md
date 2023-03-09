# Create a Truss of a TensorFlow model

[TensorFlow](https://www.tensorflow.org/) is a supported framework on Truss. To package a TensorFlow model, follow the steps below  or run [this Google Colab notebook](https://colab.research.google.com/github/basetenlabs/truss/blob/main/docs/notebooks/tensorflow_example.ipynb).

### Install packages

If you're using a Jupyter notebook, add a line to install the `tensorflow` and `truss` packages. Otherwise, ensure the packages are installed in your Python environment.

```python
!pip install --upgrade tensorflow truss
# For help installing tensorflow, see https://www.tensorflow.org/install/pip
```

{% hint style="warning" %}
Truss officially supports `tensorflow` version 2.4.0 or higher. Especially if you're using an online notebook environment like Google Colab or a bundle of packages like Anaconda, ensure that the version you are using is supported. If it's not, use the `--upgrade` flag and pip will install the most recent version.
{% endhint %}

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

Use the `create` command to package your model into a Truss.

```python
from truss import create

tr = create(model, target_directory="tensorflow_truss")
```

Check the target directory to see your new Truss!

### Serve the model

In your newly created Truss, open `model/model.py` and add pre- and post-processing functions as follows.

First, add the following imports at the top of the file:


```python
import requests
import tempfile
import tensorflow as tf
from scipy.special import softmax
```

Then, update the pre-processing function to:

```python
def preprocess(self, model_input: Any) -> Any:
    """Preprocess step for ResNet"""
    request = requests.get(model_input)
    with tempfile.NamedTemporaryFile() as f:
        f.write(request.content)
        f.seek(0)
        input_image = tf.image.decode_png(tf.io.read_file(f.name))
    preprocessed_image = tf.keras.applications.resnet_v2.preprocess_input(
        tf.image.resize([input_image], (224, 224))
    )
    return np.array(preprocessed_image)
```

Finally, update the post-processing function to:

```python
def postprocess(self, model_output: Dict, k=5) -> Dict:
    """Post process step for ResNet"""
    class_predictions = model_output["predictions"][0]
    LABELS = requests.get(
        "https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt"
    ).text.split("\n")
    class_probabilities = softmax(class_predictions)
    top_probability_indices = class_probabilities.argsort()[::-1][:k].tolist()
    return {
        LABELS[index]: 100 * class_probabilities[index].round(3)
        for index in top_probability_indices
    }
```

With these functions in place, you can invoke the model and pass it a URL, as in:

```python
from truss import load

tr = load("tensorflow_truss")
tr.predict("https://github.com/pytorch/hub/raw/master/images/dog.jpg")
```

For information on running the Truss locally, see [local development](../develop/localhost.md).
