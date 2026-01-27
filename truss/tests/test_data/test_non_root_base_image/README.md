Test case for building a truss even when the base image has a `nonroot` UID.

VERSION=1. Bump + replace below every time we push a new image.

```
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  --push \
  -f ./Dockerfile \
  -t baseten/truss-integration-tests:non-root-base-image-v{VERSION} \
  . || { echo "Failed to build and push the Docker image"; exit 1; }
```
