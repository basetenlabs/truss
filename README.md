# Truss

**Serve any model without boilerplate code**

![Truss logo](docs/assets/logo.png)

Meet Truss, a seamless bridge from model development to model delivery. Truss presents an open-source standard for packaging models built in any framework for sharing and deployment in any environment, local or production.

## Use cases

List here, maybe with emoji or something

What is Truss useful for?

* Let's do best practices and not do boilerplate
    * Freezing dependencies. Making the environment that you trained your model in very portable. Accomplished using Docker. Now your model can be used and run elsewhere.
    * Giving you an API. Turns a Python object into a web object. Now all the people that speak web can internet it.
    * Not thinking about: supported framework serialization and deserialization, in all normal circumstances but long tail, model serving framework (Flask, Django, FastAPI, Tornado)
    * Delegate responsibility to people who know what they are doing
* Exposes just the right amount of complexity around things like docker and APIs without you having to really think about them

How do you imagine using Truss?

* Truss for local API development with React frontend

Truss = Model backend

Model as a microservice

Every model runs in its own environment (different intuition than 5 CRUD API endpoints)

What projects are you excited to see people build with Truss?

New model, in whatever framework, and I should just be able to run it on some web server. Model exists and I can run it are equivalent statements. Reliably work to package this thing so it is ready for the web and all the different ways people want to interface with it.

Models take numbers. Embedding is a function that turns text to numbers

React-like iterative functionality. Right now Truss is 0-1. Eventual iterative features around anomoly detection, drift, etc. Or have multiple trusses talk to each other. Separate model explains output.

Truss makes your ML model "atomic" ... it makes the ML model a single block within your system... MaaM

Truss is a single-celled organism. Is it a Prokaryote or Eucaryote? 


ROADMAP

## Quickstart

Generate and serve predictions from a Truss with this Jupyter notebook

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

Though Truss is in beta, we do care about backward compatibility. Review the [release notes](docs/CHANGELOG.md) before upgrading, and note that we follow semantic versioning, so any breaking changes require the release of a new major version.

## Truss' purpose



## Contributions

Truss was first developed at [Baseten](https://baseten.co) by Phil Howes, Pankaj Gupta, and Alex Gillmor.

We gratefully welcome contributions in accordance with our [contributors' guide](CONTRIBUTING.md) and [code of conduct](CODE_OF_CONDUCT.md).