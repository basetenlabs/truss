# Truss

**Serve any model without boilerplate code**

![Truss logo](docs/assets/logo.png)

Meet Truss, a seamless bridge from model development to model delivery. Truss presents an open-source standard for packaging models built in any framework for sharing and deployment in any environment, local or production.

## Quickstart

Generate and serve predictions from a Truss with [this Jupyter notebook]().

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
# TRUSS PREDICT
```

## Use cases

Truss exposes just the right amount of complexity around things like Docker and APIs without you really having to think about them. 

* Freezes dependencies via Docker to make your training environment portable
* Turns your Python model into a microservice with a production-ready API endpoint, no need for Flask or Django
* For most popular frameworks, automatic model serialization and deserialization
* Rapid iteration with local development that matches your production environment
* Ship parsing and even business logic alongside your model with integrated pre- and post-processing functions
* GPU support (currently limited to certain hardware, more coming soon)
* Secret management to securely give your model access to API keys

### Model as an API


Truss = Model backend

Model as a microservice

Every model runs in its own environment (different intuition than 5 CRUD API endpoints)

Truss makes your ML model "atomic" ... it makes the ML model a single block within your system... MaaM

Truss is a single-celled organism. Is it a Prokaryote or Eucaryote? 



### Model as a sharable artifact

New model, in whatever framework, and I should just be able to run it on some web server. Model exists and I can run it are equivalent statements. Reliably work to package this thing so it is ready for the web and all the different ways people want to interface with it.

Models take numbers. Embedding is a function that turns text to numbers

React-like iterative functionality. Right now Truss is 0-1. Eventual iterative features around anomoly detection, drift, etc. Or have multiple trusses talk to each other. Separate model explains output.

Truss's use cases are expanding quickly, ROADMAP

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

## Our vision for Truss

Data Scientists and machine learning engineers spend a lot of time building models that solve specific problems: sentiment classification, facial recognition, anomaly detection. Generally, this exploratory, experimental work is done using an interactive environment like a Jupyter notebook. To match the wide variety of use cases for ML models, there are a number of popular frameworks to build in, like PyTorch, TensorFlow, and scikit-learn. Each framework specializes in different kinds of models, so a data scientist will pick the framework based on the type of problem they are solving.

As such, the data scientist's development environment needs to be flexible and permissive. Jupyter notebooks are a great tool for training models, but as a impermanent and development-oriented environment they aren't great for model serving. Model serving, or making a model available to other systems, is critical; a model is not very useful unless it can operate in the real world.

There are many ways to serve a model, but most involve the following steps in some form:

1. Serialize the model
2. Put the model behind a web server such as Flask
3. Package the web server into a Docker image
4. Run the Docker image on a container

DevOps is its own specialty for a reason: the endless configuration options and compatibility checks throughout this process can be overwhelming. So Data Scientists, if they're lucky, are able to turn to infrastructure teams for help. But this is not an ideal solution:

* Data scientists compete for the limited bandwidth of infrastructure teams, leading to long wait times to get models deployed.
* The cost and friction of accessing infrastructure expertise means that only the safest ideas ever see the light of day. A lot of brilliant models may not seem promising at first, and will die in the backlog before reaching their potential.
* Debugging is hard when the model serving environment is different from the data scientist's notebook, introducing cumbersome and time-consuming effort to get everything set up right.

Truss bridges the gap between model development and model deployment by making it equally straightforward to serve a model in localhost and in prod, making development and testing loops rapid.

We built and open-sourced Truss with the conviction that eliminating this friction will accelerate machine learning productivity:

* Data scientists can build or deploy Docker images with a single command, reducing the model serving workload.
* Models can be packaged in a standardized format, making it easier to share models within or beyond a team.
* Data scientists can build on each other's work by pulling down popular models without spending hours coaxing them into running in a new environment.
* Providing a web-first interface for models will encourage real-world use and new applications.

## Contributing

We hope this vision excites you, and we gratefully welcome contributions in accordance with our [contributors' guide](CONTRIBUTING.md) and [code of conduct](CODE_OF_CONDUCT.md).

Truss was first developed at [Baseten](https://baseten.co) by maintainers Phil Howes, Pankaj Gupta, and Alex Gillmor.