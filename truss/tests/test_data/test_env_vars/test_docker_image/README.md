We built this minimal fastapi docker image to be used in integration test `test_custom_server_truss.py::test_custom_server_truss`

Steps to update testing docker image

1. run `docker login`
2. cd into this directory
3. update version number in VERSION file
(before running the next step, make sure you meet with the [prerequisites](https://docs.docker.com/build/building/multi-platform/#prerequisites) here)
4. run `sh build_upload_new_image.sh`
5. update image tag to latest version in config.yaml
