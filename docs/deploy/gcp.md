# Deploy a Truss to GCP

In this guide, we'll cover how to deploy a Truss to GCP Cloud Run.

Prerequisites:

* A GCP account with appropriate access
* [GCloud SDK](https://cloud.google.com/sdk/docs/install)

In this example, we'll be deploying the [TensorFlow Truss](../create/tensorflow.md) from an earlier tutorial. If you don't already have your Truss, use that tutorial to make one.

First, we'll set a couple of important values for the Docker container.

```python
from pathlib import Path
SERVICE_NAME = "tensorflow-truss-model"
TARGET_TRUSS_BUILD_DIRECTORY = Path("tensorflow_truss_build")
```

Then, we build the Docker container.

```python
import truss

truss.docker_build_setup(build_dir=TARGET_TRUSS_BUILD_DIRECTORY)
```

### Configure GCP

Enable the following three APIs:

1. Cloud Run API
2. Artifact Registry API
3. Cloud Build API

Then deploy your model from the terminal!

```
gcloud run deploy tensorflow-truss-model --source tensorflow_truss_build --allow-unauthenticated --memory 8GiB
```

If you get the following error:

```
INVALID_ARGUMENT: could not resolve source: googleapi: Error 403: XXXXXXXXXXX@cloudbuild.gserviceaccount.com does not have storage.objects.get access to the Google Cloud Storage object
```

Re-run the command and enable the APIs through the command line following [this GCP tutorial](https://cloud.google.com/endpoints/docs/openapi/enable-api).
