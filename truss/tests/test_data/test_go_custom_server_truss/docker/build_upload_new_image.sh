# Read the version from the VERSION file
VERSION=$(cat VERSION)

docker buildx build \
  --platform linux/amd64,linux/arm64 \
  --push \
  -f ./Dockerfile \
  -t baseten/go-custom-server-test:$VERSION \
  . || { echo "Failed to build and push the Docker image"; exit 1; }
