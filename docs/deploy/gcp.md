# Deploy a Truss to GCP

In this guide, we'll cover how to deploy a Truss docker image to GCP Cloud Run.

Prerequisites:

* A GCP account with appropriate access
* [GCloud SDK](https://cloud.google.com/sdk/docs/install)


1. Set up 

2. Enable:

    1) Cloud Run API 
    2) Artifact Registry API
    3) Cloud Build API. Wait a few minutes, if you get: 
    
        INVALID_ARGUMENT: could not resolve source: googleapi: Error 403: XXXXXXXXXXX@cloudbuild.gserviceaccount.com does not have storage.objects.get access to the Google Cloud Storage object

    in your GCP Project. Runing the command below, also gives you the option to enable these APIs through the command line! (https://cloud.google.com/endpoints/docs/openapi/enable-api)