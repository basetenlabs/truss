# Pre- and post-processing

When you serialize a model, you want to keep the output as possible, containing just the model and its direct dependencies without custom logic or extra frameworks. This is great for keeping file sizes small and deserialization fast, but makes it tough to know what to do with the serialized model.

One feature that makes Truss different from model serialization frameworks is bundled pre- and post-processing. Most models intended for use in production systems will want to have these functions as part of the Truss.

Add pre-processing and post-processing functions to your Truss by editing `model/model.py` to include the necessary functions.

## Pre-processing

ML models are picky eaters. One of the difficulties of using a model in a production system is that it may only take a certain type of input, specifically formatted. This means anyone and anything interacting with the model needs to be aware of the nuances of its input requirements.

Truss allows you to instead bundle pre-processing Python code with the model itself. This code is run on every call to the model before the model itself is run, giving you the chance to define your own input format.

Here is a pre-processing function from the [TensorFlow example](../create/tensorflow.md) that downloads an image from a URL and sizes it for the ResNet model.

```python
import requests
import tempfile
import numpy as np

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

## Post-processing

Similarly, the output of a ML model can be messy or cryptic. Demystify the model results for your end user or format a web-friendly response in the post-processing function.

Here is a post-processing function from the same [TensorFlow example](../create/tensorflow.md) that returns clean labels.

```python
from scipy.special import softmax

def postprocess(self, model_output: Any, k=5) -> Any:
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
