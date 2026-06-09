#!/usr/bin/env sh
set -eu

VERSION="$(cat VERSION)"
REMOTE_IMAGE_NAME="baseten/performance-client-integration-truss-server:${VERSION}"

docker buildx build \
  --platform linux/amd64 \
  -f ./Dockerfile \
  -t "${REMOTE_IMAGE_NAME}" \
  --push \
  . || {
  echo "Failed to build and push the Docker image"
  exit 1
}
