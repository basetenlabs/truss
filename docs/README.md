# Truss documentation

**Serve any model without boilerplate code**

Truss logo coming soon

Meet Truss, a seamless bridge from model development to model delivery. Truss presents an open-source standard for packaging models built in any framework for sharing and deployment in any environment, local or production.

## Capabilities

TODO

## Quickstart

Generate and serve predictions from a Truss:

```
!pip install scikit-learn
!pip install truss
import truss
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

iris = load_iris()
data_x = iris['data']
data_y = iris['target']
rfc = RandomForestClassifier()
rfc.fit(data_x, data_y)

truss.mk_truss(rfc, target_directory="iris_rfc")
```

## Installation

Truss requires Python >= 3.7,<3.11

To install from [PyPi](https://pypi.org/project/truss/), run:

```
pip install truss
```

To download the source code directly (for development), clone this repository and follow the setup commands in our [contributors' guide](CONTRIBUTING.md).

Truss is actively developed, and we recommend using the latest version. To update your Truss installation, run:

```
pip install --upgrade truss
```

Though Truss is in beta, we do care about backward compatibility. Review the [release notes](CHANGELOG.md) before upgrading, and note that we follow semantic versioning, so any breaking changes require the release of a new major version.

## Our vision for Truss

Machine learning is about building systems that learn from data rather than execute explicit instructions. Data Scientists and ML engineers spend a lot of time building models for solving specific problems such as facial recognition and anomaly detection. They typically do this work using interactive environments such as jupyter notebooks. There are many frameworks for facilitating machine learning. PyTorch, tensorflow and sklearn are some of the popular ones, but there are many more. ML Devs would typically use one of the frameworks to develop their model in. Further, each framework supports many different types of models and depending on the problem they would pick a model type. e.g. CNNs (Convolutional Neural networks) are popular for image recognition, RNNs for time series prediction and random forests for classification. Once ML Devs have developed their model in their notebook, they would prepare it for serving. Model serving is critical, a model is not very useful unless it can be put to real world use.

There are many ways of serving a model, but most involve the following steps in some form:

1. Serialize the model
2. Put the model behind a web server such as flask
3. Package the webserver into a docker image
4. Run the docker image on a container


All these choices can be overwhelming for an ML Dev and they typically have infra teams help them out with it. But this is not an ideal solution for many reasons:

ML Devs have to fight for attention from the infra team, which is typically always overwhelmed. They often have to wait weeks if not months.
Cost of access to infra expertise makes trying things harders. Only the most promising ideas see the light of day. A lot of good ideas may not seem very promising at first, for lack of trying they would die in the backlog.
Debugging can be hard, the model serving environment may be different from their notebook environment. The human back and forth to get the set up right is cumbersome and time consuming.
Scaffolds bridges this gap by making it straight forward to prepare their model for serving. It allows them to quickly try out the model server directly on their laptop, making the model serving development loop quick and smooth.

With scaffolds, users can simply pass their model reference to a function to generate a model package. They can then run this model locally, or build a docker image out of it or deploy to baseten, all with a single command each.

They can also pull down any of the many popular models that we have in the model zoo, modify them locally and deploy for themselves.

## Contributions

Truss was first developed at [Baseten](https://baseten.co) by Phil Howes, Pankaj Gupta, and Alex Gillmor.

We gratefully welcome contributions in accordance with our [contributors' guide](../CONTRIBUTING.md) and [code of conduct](../CODE_OF_CONDUCT.md).