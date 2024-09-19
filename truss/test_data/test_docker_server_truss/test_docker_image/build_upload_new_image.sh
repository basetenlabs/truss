# Read the version from the VERSION file
VERSION=$(cat VERSION)

docker build -t fastapi-test .
docker tag fastapi-test baseten/fastapi-test:$VERSION
docker push baseten/fastapi-test:$VERSION
