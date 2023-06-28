# Pre- and post-processing

An efficiently serialized model is just the model and its direct dependencies. No custom logic or extra frameworks. This makes deserialization fast, but may make invoking the model inconvenient.

Write pre- and post-processing functions to:

* Format model inputs and outputs, especially when the model needs something like an `xgboost.DMatrix` that isn't JSON-serializable
* Add custom logic like saving model outputs to a data store
* Factor out unnecessary logic from `predict()`

{% hint style="warning"%}
There is a GPU lock during `predict()` but not `preprocess()` and `postprocess()`. That means:

* If you're interacting with the GPU, do it during `predict()` to avoid putting the GPU in a bad state.
* If you're doing any long-running calculation not on a GPU (e.g. making a HTTP request), do it in a processing function so the GPU can be freed up for the next request.
{% endhint %}

### Pre-processing

ML models are picky eaters. One of the difficulties of using a model in a production system is that it may only take a certain type of input, specifically formatted. This means anyone and anything interacting with the model needs to be aware of the nuances of its input requirements.

Truss allows you to instead bundle pre-processing Python code with the model itself. This code is run on every call to the model before the model itself is run, giving you the chance to define your own input format.

**Example: ResNet pre-processing**

Here is a pre-processing function from the [TensorFlow example](../create/tensorflow.md) that downloads an image from a URL and sizes it for the ResNet model.

```python
import requests
import tempfile
import numpy as np

class Model:

    def __init__(self, **kwargs):
        ...

    def load(self):
        ...

    def predict(self, model_input):
        ...

    def preprocess(self, model_input):
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

### Post-processing

Similarly, the output of a ML model can be messy or cryptic. Demystify the model results for your end user or format a web-friendly response in the post-processing function.

**Example: ResNet post-processing**

Here is a post-processing function from the same [TensorFlow example](../create/tensorflow.md) that returns clean labels.

```python
from scipy.special import softmax

class Model:

    def __init__(self, **kwargs):
        ...

    def load(self):
        ...

    def predict(self, model_input):
        ...

    def postprocess(self, model_output):
        class_predictions = model_output["predictions"][0]
        LABELS = requests.get(
            "https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt"
        ).text.split("\n")
        class_probabilities = softmax(class_predictions)
        top_probability_indices = class_probabilities.argsort()[::-1][:5].tolist()
        return {
            LABELS[index]: 100 * class_probabilities[index].round(3)
            for index in top_probability_indices
        }
```

## Using processing functions

Once you've modified your processing functions, you can test them by reloading the Truss and invoking the model.

```python
tr = truss.load("my-truss")
tr.predict("MODEL_INPUT")
```
