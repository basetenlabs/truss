We built this minimal fastapi docker image to be used in integration test `test_model_inference.py::test_docker_server_truss`

Steps to update testing docker image

1. run `docker login`
2. cd into this directory
3. update version number in VERSION file
3. run `sh build_upload_new_image.sh`
4. update image tag to latest version in config.yaml
