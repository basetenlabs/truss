# Local model serving

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
tr = truss.from_directory("path_to_my_truss")
```

From there, you can invoke the Truss to serve the model in your Python environment. Just run:

```python
tr.docker_predict({"inputs": [0, 0, 0, 0]})
```

### Command line interface

Alternately, you can run your Truss from the command line.

```
truss predict path_to_my_truss '{"inputs": [0, 0, 0, 0]}'
```

You can also specify examples for the model and run them instead. It's much easier to express request data in the example file. Running the example provides for a good dev loop.

### On using Docker

Testing through a Docker image closely simulates the production serving environment and is great for final testing. But it could be too slow for a tight dev loop. For a faster dev loop you can run prediction on the Truss directory directly.

Unlike Docker image, this mechanism requires that you already have the right Python requirements and system packages installed.

In the Python environment, get a prediction without Docker by running:

```python
tr.server_predict({"inputs": [0, 0, 0, 0]})
```

Or in the command line, run:

```python
truss predict --run-local path_to_my_truss '{"inputs": [0, 0, 0, 0]}'
```
