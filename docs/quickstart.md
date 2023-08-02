---
description: Use Truss to package a model.
---

# Quickstart

In this doc, we'll package a text classification model. We're going to use a [text classification pipeline](https://huggingface.co/docs/transformers/main_classes/pipelines) from the open-source [`transformers` package](https://github.com/huggingface/transformers), which includes many pre-trained models.

We'll use [Truss](https://trussml.com), an open-source model packaging library maintained by [Baseten](https://baseten.co), to package the model.

### Create a Truss

To get started, create a Truss with the following terminal command:

```
truss init text-classification
```

This will create an empty Truss at `./text-classification`.

### Implement the model

The model serving code goes in `./text-classification/model/model.py` in your newly created Truss.

```python
from typing import List
from transformers import pipeline


class Model:
    def __init__(self, **kwargs) -> None:
        self._model = None

    def load(self):
        self._model = pipeline("text-classification")

    def predict(self, model_input: str) -> List:
        return self._model(model_input)
```

There are two functions to implement:

* `load()` runs once when the model is spun up and is responsible for initializing `self._model`
* `predict()` runs each time the model is invoked and handles the inference. It can use any JSON-serializable type as input and output.

### Add model dependencies

The pipeline model relies on Transformers and PyTorch. These dependencies must be specified in the Truss config.

In `./text-classification/config.yaml`, find the line `requirements`. Replace the empty list with:

```yaml
requirements:
  - torch==2.0.1
  - transformers==4.30.0
```

No other configuration needs to be changed.

You've successfully packaged a model! Next, [deploy it](deploy/baseten.md).
