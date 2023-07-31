# Changelog

Release notes for new versions of Truss, in reverse chronological order.

See [https://github.com/basetenlabs/truss/releases/](https://github.com/basetenlabs/truss/releases/)

### Version 0.4.8

Base image improvements:

* Updated base images now use Ubuntu 20.04 and support new Python versions (Truss supports Python 3.8 through Python 3.11)
* GPU base images now have python#.#-dev to support more Python libraries
* Both draft and published models can now use custom base images

Truss package cleanup:

* Dropped kserve as the base server framework

Model weight and data speedups:

* Model weight downloads for Whisper and other open-source OpenAI models are now concurrent
* External data is now packaged at build time instead of at runtime

### Version 0.4.0

This release updates the model invocation interface in Truss templates, affecting **only newly created Trusses**.

Until now, when creating a Truss, the `predict()` function expected a dictionary with the key `inputs` and a value to give the model as input. This default behavior was due to a legacy requirement in Baseten that was removed months ago.

Now, the default behavior is to use the input of `predict()` directly, which can be anything JSON-serializable: string, number, list, dictionary, etc. If you want, you can modify the `predict()` function to expect a dictionary with `inputs` like the old spec.

Before (sklearn iris):

```python
model.predict({"inputs": [[0, 0, 0, 0]]})
```

Now (sklearn iris):

```python
model.predict([[0, 0, 0, 0]])
```

**This change does not affect existing Trusses.** Only new Trusses created on this version (i.e. by running `truss.create()` or `truss init`) will use the updated templates.

### Version 0.3.0

This version was created to support [blueprint](https://blueprint.baseten.co).

### Version 0.2.0

With this release, a minor version increment recognizes the overall progress made on Truss since its initial release in Summer 2022. And simplified naming for key functions improves Truss' developer experience, while carefully considered warnings and a long deprecation period ensure nothing breaks.

#### Interface changes

* In the Python client, `truss.create()` replaces `truss.mk_truss()`.
* In the Python client, `truss.load()` replaces `truss.from_directory()`.
* In the Truss handle, `truss.predict()` offers a shorter alternative to `truss.server_predict()`. To use in place of `truss.docker_predict()`, pass the optional kwarg `use_docker=True`.
* In the command-line interface, the behavior of `truss predict` has been updated to match the Python client.
  * Previously, `truss predict` ran on Docker by default, which could be overriden with `RUN_LOCAL=true`.
  * Now, `truss predict` runs without Docker by default, which can be overriden with `USE_DOCKER=true`.

These interface changes are intended to improve Truss' developer experience, not cause unnecessary trouble. As such, the old `mk_truss()` and `from_directory()` functions, while marked with a deprecation warning, will not be removed until the next major version update. And both `server_predict()` and `docker_predict()` will be supported in the Truss handle indefinitely.

### Version 0.1.5

This release adds the `live_reload` option. This feature makes it faster to run a Truss in Docker in some situations, letting you develop your Truss without waiting for Docker to rebuild between changes.

With this release, the live reload option supports changes to model code. In the future, we will support other changes, like changes to environment variables and Python dependencies.

### Version 0.1.4

This release adds support for MLflow. Package your MLflow model by following [this documentation](create/mlflow.md).

### Version 0.1.3

This release patches a bug from 0.1.2, no new features.

### Version 0.1.2

This release adds support for more flexible model-to-truss method (via in-memory function).

### Version 0.1.1

This release:

* Fixes inference in iPython environments
* Prints Truss handle errors in notebooks
* Adds `spec_version` flag (only relevant for using Truss with Baseten)
* Improves codespace developer experience

### Version 0.1.0 (initial release)

This release introduces Truss, and as such everything is new!
