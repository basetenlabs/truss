# Changelog

Release notes for new versions of Truss, in reverse chronological order.

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
