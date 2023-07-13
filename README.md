# Truss

**The simplest way to serve AI/ML models in production**

[![PyPI version](https://badge.fury.io/py/truss.svg)](https://badge.fury.io/py/truss)
[![ci_status](https://github.com/basetenlabs/truss/actions/workflows/release.yml/badge.svg)](https://github.com/basetenlabs/truss/actions/workflows/release.yml)

## Why Truss?

* **Write once, run anywhere:** Package and test model code, weights, and dependencies with a model server that behaves the same in development and production.
* **Fast developer loop:** Implement your model with fast feedback from a live reload server, and skip Docker and Kubernetes configuration with Truss' done-for-you model serving environment.
* **Support for all Python frameworks**: From `transformers` and `diffusors` to `PyTorch` and `Tensorflow` to `XGBoost` and `sklearn`, Truss supports models created with any framework, even entirely custom models.

See Trusses for popular models including:

* 🦅 [Falcon 40B](https://github.com/basetenlabs/falcon-40b-truss)
* 🧙 [WizardLM](https://github.com/basetenlabs/wizardlm-truss)
* 🎨 [Stable Diffusion](https://github.com/basetenlabs/stable-diffusion-truss)
* 🗣 [Whisper](https://github.com/basetenlabs/whisper-truss)

and [dozens more examples](examples/).

## Installation

Install Truss with:

```
pip install --upgrade truss
```

## Quickstart

As a quick example, we'll package a [text classification pipeline](https://huggingface.co/docs/transformers/main_classes/pipelines) from the open-source [`transformers` package](https://github.com/huggingface/transformers).

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

## Running

You can run a Truss server locally by:

```
cd ./text-classification
truss run-image
```

## Truss contributors

Truss is backed by Baseten and built in collaboration with ML engineers worldwide. Special thanks to [Stephan Auerhahn](https://github.com/palp) @ [stability.ai](https://stability.ai/) and [Daniel Sarfati](https://github.com/dsarfati) @ [Salad Technologies](https://salad.com/) for their contributions.

We enthusiastically welcome contributions in accordance with our [contributors' guide](CONTRIBUTING.md) and [code of conduct](CODE_OF_CONDUCT.md).
