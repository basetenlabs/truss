# Truss

**Serve any model without boilerplate code**

![Truss logo](https://raw.githubusercontent.com/basetenlabs/truss/main/docs/assets/truss_logo_horizontal.png)

[![PyPI version](https://badge.fury.io/py/truss.svg)](https://badge.fury.io/py/truss)
[![ci_status](https://github.com/basetenlabs/truss/actions/workflows/main.yml/badge.svg)](https://github.com/basetenlabs/truss/actions/workflows/main.yml)

Meet Truss, a seamless bridge from model development to model delivery. Truss presents an open-source standard for packaging models built in any framework for sharing and deployment in any environment, local or production.

Get started with the [end-to-end tutorial](https://truss.baseten.co/e2e).

## What can I do with Truss?

If you've ever tried to get a model out of a Jupyter notebook, Truss is for you.

Truss exposes just the right amount of complexity around things like Docker and APIs without you really having to think about them. Here are some of the things Truss does:

* 🏎 Turns your Python model into a microservice with a production-ready API endpoint, no need for Flask or Django.
* 🎚 For most popular frameworks, includes automatic model serialization and deserialization.
* 🛍 Freezes dependencies via Docker to make your training environment portable.
* 🕰 Enables rapid iteration with local development that matches your production environment.
* 🗃 Encourages shipping parsing and even business logic alongside your model with integrated pre- and post-processing functions.
* 🤖 Supports running predictions on GPUs. (Currently limited to certain hardware, more coming soon)
* 🙉 Bundles secret management to securely give your model access to API keys.

## Installation

Truss requires Python >=3.7, <3.11

To install from [PyPi](https://pypi.org/project/truss/), run:

```
pip install truss
```

To download the source code directly (for development), clone this repository and follow the setup commands in our [contributors' guide](CONTRIBUTING.md).

Truss is actively developed, and we recommend using the latest version. To update your Truss installation, run:

```
pip install --upgrade truss
```

Though Truss is in beta, we do care about backward compatibility. Review the [release notes](docs/CHANGELOG.md) before upgrading, and note that we follow semantic versioning, so any breaking changes require the release of a new major version.

## How to use Truss

Generate and serve predictions from a Truss with [this Jupyter notebook](docs/notebooks/sklearn_example.ipynb).

### Quickstart: making a Truss

```python
!pip install --upgrade scikit-learn truss

import truss
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# Load the iris data set
iris = load_iris()
data_x = iris['data']
data_y = iris['target']

# Train the model
rfc = RandomForestClassifier()
rfc.fit(data_x, data_y)

# Create the Truss (serializing & packaging model)
tr = truss.create(rfc, target_directory="iris_rfc_truss")

# Serve a prediction from the model
tr.predict({"inputs": [[0, 0, 0, 0]]})
```

### Package your model

The `truss.create()` command can be used with any supported framework:

* [Hugging Face](https://truss.baseten.co/create/huggingface)
* [LightGBM](https://truss.baseten.co/create/lightgbm)
* [PyTorch](https://truss.baseten.co/create/pytorch)
* [scikit-learn](https://truss.baseten.co/create/sklearn)
* [Tensorflow](https://truss.baseten.co/create/tensorflow)
* [XGBoost](https://truss.baseten.co/create/xgboost)

But in more complex cases, you can build a Truss manually for any model. Start with `truss init my_truss` and follow [this guide](https://truss.baseten.co/create/manual).

### Serve your model locally

Serving your model with Truss, on Docker, lets you interface with your model via HTTP requests. Start your model server with:

```
truss run-image iris_rfc_truss
```

Then, as long as the container is running, you can invoke the model as an API as follows:

```
curl -X POST http://127.0.0.1:8080/v1/models/model:predict -d '{"inputs": [[0, 0, 0, 0]]}'
```

### Configure your model for deployment

Truss is configurable to its core. Every Truss must include a file `config.yaml` in its root directory, which is automatically generated when the Truss is created. However, configuration is optional. Every configurable value has a sensible default, and a completely empty config file is valid.

The Truss we generated above in the quickstart sample has a good example of a typical Truss config:

```yaml
model_framework: sklearn
model_metadata:
  model_binary_dir: model
  supports_predict_proba: true
python_version: py39
requirements:
- scikit-learn==1.0.2
- threadpoolctl==3.0.0
- joblib==1.1.0
- numpy==1.20.3
- scipy==1.7.3
```

Follow the [configuration guide](https://truss.baseten.co/develop/configuration) and use the complete reference of configurable properties to make your Truss perform exactly as you wish.

### Deploy your model

You can deploy a Truss anywhere that can run a Docker image, as well as purpose-built platforms like [Baseten](https://baseten.co).

Follow step-by-step deployment guides for the following platforms:

* [AWS ECS](https://truss.baseten.co/deploy/aws)
* [Baseten](https://truss.baseten.co/deploy/baseten)
* [GCP Cloud Run](https://truss.baseten.co/deploy/gcp)

## Contributing

We hope this vision excites you, and we gratefully welcome contributions in accordance with our [contributors' guide](CONTRIBUTING.md) and [code of conduct](CODE_OF_CONDUCT.md).

Truss was first developed at [Baseten](https://baseten.co) by maintainers Phil Howes, Pankaj Gupta, and Alex Gillmor.

## GitHub Codespace

If your organization allows to access to GitHub Codespaces, you can launch a Codespace for truss development. If you are a GPU Codespace, make sure to use the `.devcontainer/gpu/devcontainer.json` configuration to have access to a GPU and be able to use it in Docker with truss.
