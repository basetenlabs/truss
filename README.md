# Truss

**The simplest way to serve AI/ML models in production**

[![PyPI version](https://badge.fury.io/py/truss.svg)](https://badge.fury.io/py/truss)
[![ci_status](https://github.com/basetenlabs/truss/actions/workflows/release.yml/badge.svg)](https://github.com/basetenlabs/truss/actions/workflows/release.yml)

## Why Truss?

* **Write once, run anywhere:** Package and test model code, weights, and dependencies with a model server that behaves the same in development and production.
* **Fast developer loop:** Implement your model with fast feedback from a live reload server, and skip Docker and Kubernetes configuration with a batteries-included model serving environment.
* **Support for all Python frameworks**: From `transformers` and `diffusers` to `PyTorch` and `TensorFlow` to `TensorRT` and `Triton`, Truss supports models created and served with any framework.

See Trusses for popular models including:

* 🦙 [Llama 2 7B](https://github.com/basetenlabs/truss-examples/tree/main/llama/llama-2-7b-chat) ([13B](https://github.com/basetenlabs/truss-examples/tree/main/llama/llama-2-13b-chat)) ([70B](https://github.com/basetenlabs/truss-examples/tree/main/llama/llama-2-70b-chat))
* 🎨 [Stable Diffusion XL](https://github.com/basetenlabs/truss-examples/tree/main/stable-diffusion/stable-diffusion-xl-1.0)
* 🗣 [Whisper](https://github.com/basetenlabs/truss-examples/tree/main/whisper/whisper-truss)

and [dozens more examples](https://github.com/basetenlabs/truss-examples/).

## Installation

Install Truss with:

```
pip install --upgrade truss
```

## Quickstart

As a quick example, we'll package a [text classification pipeline](https://huggingface.co/docs/transformers/main_classes/pipelines) from the open-source [`transformers` package](https://github.com/huggingface/transformers).

### Create a Truss

To get started, create a Truss with the following terminal command:

```sh
truss init text-classification
```

When prompted, give your Truss a name like `Text classification`.

Then, navigate to the newly created directory:

```sh
cd text-classification
```

### Implement the model

One of the two essential files in a Truss is `model/model.py`. In this file, you write a `Model` class: an interface between the ML model that you're packaging and the model server that you're running it on.

There are two member functions that you must implement in the `Model` class:

* `load()` loads the model onto the model server. It runs exactly once when the model server is spun up or patched.
* `predict()` handles model inference. It runs every time the model server is called.

Here's the complete `model/model.py` for the text classification model:

```python
from transformers import pipeline


class Model:
    def __init__(self, **kwargs):
        self._model = None

    def load(self):
        self._model = pipeline("text-classification")

    def predict(self, model_input):
        return self._model(model_input)
```

### Add model dependencies

The other essential file in a Truss is `config.yaml`, which configures the model serving environment. For a complete list of the config options, see [the config reference](https://truss.baseten.co/reference/config).

The pipeline model relies on [Transformers](https://huggingface.co/docs/transformers/index) and [PyTorch](https://pytorch.org/). These dependencies must be specified in the Truss config.

In `config.yaml`, find the line `requirements`. Replace the empty list with:

```yaml
requirements:
  - torch==2.0.1
  - transformers==4.30.0
```

No other configuration is needed.

## Deployment

Truss is maintained by [Baseten](https://baseten.co), which provides infrastructure for running ML models in production. We'll use Baseten as the remote host for your model.

Other remotes are coming soon, starting with AWS SageMaker.

### Get an API key

To set up the Baseten remote, you'll need a [Baseten API key](https://app.baseten.co/settings/account/api_keys). If you don't have a Baseten account, no worries, just [sign up for an account](https://app.baseten.co/signup/) and you'll be issued plenty of free credits to get you started.

### Run `truss push`

With your Baseten API key ready to paste when prompted, you can deploy your model:

```sh
truss push
```

You can monitor your model deployment from [your model dashboard on Baseten](https://app.baseten.co/models/).

### Invoke the model

After the model has finished deploying, you can invoke it from the terminal.

**Invocation**

```sh
truss predict -d '"Truss is awesome!"'
```

**Response**

```json
[
  {
    "label": "POSITIVE",
    "score": 0.999873161315918
  }
]
```

## Truss contributors

Truss is backed by Baseten and built in collaboration with ML engineers worldwide. Special thanks to [Stephan Auerhahn](https://github.com/palp) @ [stability.ai](https://stability.ai/) and [Daniel Sarfati](https://github.com/dsarfati) @ [Salad Technologies](https://salad.com/) for their contributions.

We enthusiastically welcome contributions in accordance with our [contributors' guide](CONTRIBUTING.md) and [code of conduct](CODE_OF_CONDUCT.md).
