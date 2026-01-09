# Truss

**The simplest way to serve AI/ML models in production**

[![PyPI version](https://badge.fury.io/py/truss.svg)](https://badge.fury.io/py/truss)
[![ci_status](https://github.com/basetenlabs/truss/actions/workflows/release.yml/badge.svg)](https://github.com/basetenlabs/truss/actions/workflows/release.yml)

Truss is an open-source framework for packaging and deploying ML models. Write
your model's serving logic in Python, and Truss handles containerization,
dependency management, and GPU configuration.

Deploy models from any framework: `transformers`, `diffusers`, PyTorch, TensorFlow, vLLM, SGLang, TensorRT-LLM, and more:

* 🦙 [Llama 4 Scout](https://github.com/basetenlabs/truss-examples/tree/main/llama/llama-4-scout-17b-16e-instruct-bf16-vllm) · [Llama 3](https://github.com/basetenlabs/truss-examples/tree/main/llama/llama-3-8b-instruct) ([70B](https://github.com/basetenlabs/truss-examples/tree/main/llama/llama-3-70b-instruct))
* 🧠 [DeepSeek R1](https://github.com/basetenlabs/truss-examples/tree/main/deepseek) — reasoning models
* 🎨 [FLUX.1](https://github.com/basetenlabs/truss-examples/tree/main/flux) — image generation
* 🗣 [Whisper v3](https://github.com/basetenlabs/truss-examples/tree/main/whisper/whisper-v3-truss) — speech recognition

**[Get started](https://docs.baseten.co/examples/deploy-your-first-model)** | [100+ examples](https://github.com/basetenlabs/truss-examples/) | [Documentation](https://docs.baseten.co)

## Why Truss?

* **Write once, run anywhere:** Package model code, weights, and dependencies with a model server that behaves the same in development and production.
* **Fast developer loop:** Iterate with live reload, skip Docker and Kubernetes configuration, and use a batteries-included serving environment.
* **Support for all Python frameworks:** From `transformers` and `diffusers` to PyTorch and TensorFlow to vLLM, SGLang, and TensorRT-LLM, Truss supports models created and served with any framework.
* **Production-ready:** Built-in support for GPUs, secrets, caching, and autoscaling when deployed to [Baseten](https://baseten.co) or your own infrastructure.

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

Truss is maintained by [Baseten](https://baseten.co) and deploys to the [Baseten Inference Stack](https://www.baseten.co/resources/guide/the-baseten-inference-stack/), which combines optimized inference runtimes with production infrastructure for autoscaling, multi-cloud reliability, and fast cold starts.

### Get an API key

To set up the Baseten remote, you'll need a
[Baseten API key](https://app.baseten.co/settings/account/api_keys). If you
don't have a Baseten account, no worries, just
[sign up for an account](https://app.baseten.co/signup/) and you'll be issued
plenty of free credits to get you started.

### Run `truss push`

With your Baseten API key ready to paste when prompted, you can deploy your
model:

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

Truss is backed by Baseten and built in collaboration with ML engineers
worldwide. Special thanks to [Stephan Auerhahn](https://github.com/palp) @
[stability.ai](https://stability.ai/) and
[Daniel Sarfati](https://github.com/dsarfati) @
[Salad Technologies](https://salad.com/) for their contributions.

We enthusiastically welcome contributions in accordance with our
[contributors' guide](CONTRIBUTING.md) and
[code of conduct](CODE_OF_CONDUCT.md).
