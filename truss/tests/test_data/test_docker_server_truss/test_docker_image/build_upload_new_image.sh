# Read the version from the VERSION file
VERSION=$(cat VERSION)

docker buildx build --platform linux/amd64,linux/arm64 -f ./Dockerfile -t fastapi-test:$VERSION . || { echo "Failed to build the Docker image"; exit 1; }
docker tag fastapi-test:$VERSION baseten/fastapi-test:$VERSION || { echo "Failed to tag the Docker image."; exit 1; }
docker push baseten/fastapi-test:$VERSION || { echo "Failed to push the Docker image."; exit 1; }
