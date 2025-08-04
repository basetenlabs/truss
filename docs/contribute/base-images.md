# Base images

Base images are used to create Docker images for containers. Truss uses customized base images to:

* Reduce Docker image build times for model serving
* Reduce container startup time
* Re-use common parts of containers

For a general background on base images, [learn more in Docker's official documentation](https://docs.docker.com/build/building/base-images/).

## Generating base images for Truss

[This script](https://github.com/basetenlabs/truss/blob/main/bin/generate_base_images.py) can be used to generate and push
base images to Docker Hub. The script allows a lot of control over which images
to generate. To learn about all options, run:

```sh
bin/generate_base_images.py --help
```

**Examples:**

This example generates a base image for a model serving Truss with live reload and GPU support. The Truss uses Python 3.9 and has a version tag of `v0.2.2`. The generated image is named `baseten/truss-server-base:3.9-gpu-reload-v0.2.2`:

```sh
uv run bin/generate_base_images.py --live-reload y --use-gpu y --python-version 3.9 --version-tag v0.2.2 --job-type server

```

This example generates and pushes to Docker Hub all images with the version tag `v0.2.2`. This requires access to a specific Docker Hub account, which is not available publicly.

```sh
uv run bin/generate_base_images.py  --version-tag v0.2.2 --push
```

## Working with base images in Truss

This section will guide you through testing and releasing new base images for Truss.

### Testing base images

Building every base image locally to test on your local Docker takes a long time. Be selective about which base images you need to build and test based on your changes. For example, you might only need to build serving images for Python 3.9 without GPU support.

Also, we recommend using a custom version tag to keep things clean while testing.

### Publishing base images

New base images are only published when changes are released that affect base images. This prevents proliferation of identical images with different versions.

When publishing base images, their semantic versioning must match the version of Truss they were built from, even if that skips versions from the existing base images. This makes it easier to track where the code came from.

To publish a base image:

1. Increment Truss version in pyproject.toml
2. Update `/workspaces/truss/truss/contexts/image_builder/util.py::TRUSS_BASE_IMAGE_VERSION_TAG` to match this version
3. Create a PR and get it reviewed.
5. Merge the PR.
6. Upon merge, a Github Action will run that generates all the necessary base images
6. Make sure integration tests pass.

### Releasing Truss versions with new base images

If and only if your changes to Truss require a new base image, follow these steps before publishing a new version to PyPi:

1. Generate your new base images with the `generate_images` script.
2. Update `TRUSS_BASE_IMAGE_VERSION_TAG` to match the new Truss version so that the new base images are used.
3. Push the new base images before running integration tests. (The new images are not used for integration tests until the Truss library and context builder are published)
4. Increment the Truss version and publish to PyPi as normal.
