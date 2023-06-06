# Truss base images and how to work with them

Base images speed up building docker images for serving as well as training from
Trusses, by effectively building and reusing the common parts. They also help
with container startup times when these images are used. We recently started
using base images more extensively. This has implications for truss development.

## Generating base images
[This script](../../bin/generate_base_images.py) can be used to generate and push
base images to dockerhub. The script allows a lot of control over which images
to generate.

```bin/generate_base_images.py --help``` to learn about all the options.

Examples:
```
# To generate base image with gpu support, for python
# version 3.9, for model serving and with a version tag v0.2.2
poetry run bin/generate_base_images.py --use-gpu y --python-version 3.9 --version-tag v0.2.2 --job-type server
# Generates image named baseten/truss-server-base:3.9-gpu-v0.2.2
```

```
# Generate and pushing to dockerhub all images with version tag v0.2.2
# Note that this requires access to a specific dockerhub account,
# which is not available publicly.
poetry run bin/generate_base_images.py  --version-tag v0.2.2 --push
```

## Working with base images

### Testing with base images
Say you're making a change that goes into base images.
One thing you could do is build all the base images locally. Once
the images are available on your local docker, you'll be able to test them. But
building all images can take a long time, so you may want to be selective. e.g.
you may want to build only serving images for python version 3.9 and without gpu
support (gpu images take longer to build). You may also want to use a custom
version tag to keep things clean.

### Publishing base images
We don't publish new base images everytime Truss version is incremented. We only
publish them when there are changes that would have an effect on the base
images; we don't want to end up with a plethera of same base images with
different tags. But when publishing base images we do match the version to Truss
version at that point, so we know what code is in the base images. When you have
a change that needs publishing base images do the following:

1. Increment Truss version in pyproject.toml
2. Update
   `/workspaces/truss/truss/contexts/image_builder/util.py::TRUSS_BASE_IMAGE_VERSION_TAG`
   to match this version
3. Create a PR and get it reviewed.
4. Once the PR is ready to land, use the [image generation script](../../bin/generate_base_images.py)
   to build and publish base images. Make sure to use the value of TRUSS_BASE_IMAGE_VERSION_TAG
   for version tag.
5. Merge the PR
6. Make sure integration tests pass
    - Integration tests need the new images to be published, so this is a good
      test for publication as well
