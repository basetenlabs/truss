# Local development

Even with the simplified experience of deploying a Truss, deploying anything to production to test changes creates a loop that is too slow for development. So instead we use Docker to create a fast local iteration loop that closely matches the production environment.

{% hint style="warning" %}
Make sure you have [Docker installed](https://docs.docker.com/get-docker/) before running a Truss locally.
{% endhint %}

In local development, you can test and iterate on aspects of your model and environment, such as:

* Dependencies: Are the Python packages you need available and working?
* System packages: Are system-level dependencies installed with the correct version?
* Environment variables: Do you have all of your config vars and API keys securely?
* Pre- and post-processing functions: Is your model accepting and outputting well-formatted data?
* Examples: Does your model perform as expected on sample inputs?

There are two ways to interface with a Truss locally. The first is via [the Python Truss object interface](../reference/client.md#truss-use), and the second via [the command line](../reference/cli.md).

### Python client

When interacting with your Truss via the Python client, the first thing is to make sure it is in-memory. If you just created the Truss, it'll be in memory already, but if not, you'll need to load it with the following command:

```python
tr = truss.load("path_to_my_truss")
```

From there, you can invoke the Truss to serve the model in your Python environment. Just run:

```python
tr.docker_predict({"inputs": [[0, 0, 0, 0]]})
```

### Command line interface

Alternately, you can run your Truss from the command line.

```
truss predict path_to_my_truss '{"inputs": [[0, 0, 0, 0]]}'
```

You can also specify examples for the model and run them instead. It's much easier to express request data in the example file. Running the example provides for a good dev loop.

### As an API

Serving your model with Truss, on Docker, lets you interface with your model via HTTP requests. Start your model server with:

```
truss run-image path_to_my_truss
```

Then, as long as the container is running, you can invoke the model as an API as follows:

```
curl -X POST http://127.0.0.1:8080/v1/models/model:predict -d '{"inputs": [[0, 0, 0, 0]]}'
```

## Setting up local dev

You have three options for how to run your Truss locally:

* In a Docker container
* In a Docker container with live reload
* Without a Docker container in the Truss directory

### Running with Docker

This is the standard way to run a Truss. It creates a Docker image, runs the Truss, and returns a prediction. This method closely matches production environments, but results in a slow dev loop as it rebuilds the Docker container with every change to your code.

To run in Docker:

```python
tr.docker_predict({"inputs": [[0, 0, 0, 0]]})
```

### Faster dev loop with live reload

You can turn on live reload by setting the `live_reload` property to `True` in `config.yaml`. This feature makes it faster to run a Truss in Docker in some situations.

It essentially works by keeping the Docker container running as you work on your Truss. Depending on the nature of the change, it may be able to update the existing container, avoiding building a new Docker image, making for a snappy development loop. This change is called a patch. Examples are changes to:

1. Model code
2. Python requirements
3. System packages
4. Environment variables

Currently, only model code changes are supported, but we are working on supporting other types of patches.To understand patch computation it's important to understand a Truss's hash and signature.

#### Truss Hash

A Truss's hash is the sha256 hash of the Truss's content.

#### Truss Signature

A Truss's signature is a form of digest that is meant for computing changes to a
Truss from a previous state. The idea is that high level changes can be gleaned
from a short previous representation rather than the whole Truss. This avoids
storing previous versions of the Truss, drastically reducing space requirements.

Currently, the signature consists of two things:

1. All the filepaths under the Truss and their content hashes
2. Contents of `config.yaml`

The signature is expected to be just a few Kbs for most cases.

### Truss without Docker

Testing through a Docker image closely simulates the production serving environment and is great for final testing. But it could be too slow for a tight dev loop. For a faster dev loop you can run prediction on the Truss directory directly.

Unlike Docker image, this mechanism requires that you already have the right Python requirements and system packages installed.

In the Python environment, get a prediction without Docker by running:

```python
tr.predict({"inputs": [[0, 0, 0, 0]]})
```

Or in the command line, run:

```python
truss predict --run-local path_to_my_truss '{"inputs": [[0, 0, 0, 0]]}'
```
