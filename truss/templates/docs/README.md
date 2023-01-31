# Baseten Scaffold

The goal of a baseten scaffold is to provide users with a short path to having a working model serving environment while
 maintaining a flexibility for customization.

At first construction of a scaffold the user is provided with some scaffold specific run strings should they want to
interact with the environment in their local environment. A container building tool like `docker` is required to do so

## Model Framework

The following model frameworks are supported as 'base' scaffolds that work with very little friction out of the box.

* `keras`
* `lightgbm`
* `pytorch`
* `sklearn`
* `transformers`
* `xgboost`

## Code Structure

A scaffold will provide the following in the `src` directory of the build context

* `common` -  A package of `baseten` specific tooling which provide our opinions over
`kserve` model servers, as well as various utilities and exceptions,
* `server` - A python package that contains `model_framework` specific code for loading model binaries
 and serving model predictions.
* `model` - A python package designed to be a destination for any code required for model operation and model binaries.
* `data` -  A folder designed for any non-code depencies required for model serving
* `requirements.txt` - A `pip` style `requirements.txt` for user software that will be installed into
the container
* `{model_framework}-server.Dockerfile` - A `dockerfile` that describes how to build the container from the context
provided by the scaffold.

### Baseten Model

The code for this will be in `model/inference_model.py`. The logic flow is such that there are the following methods that are run
 in order

* `load` - This is run once when the process is started and is a good place to do any expensive computations or data loading that needs to be run once

With these methods running on every request in order.
* `preprocess` - This is an ideal place to place any input level feature transformed that might not be encapsulated by the model object.
* `predict` - This is the main prediction method; by default we support most JSON serializable objects in addition to numpy objects.
* `postprecess` - This is an ideal place to place any output level feature transforms that might not be encapsulated by the model.

## Custom Scaffolds

If the user requires more flexibility than provided from a 'base' scaffold they are welcome to make changes to the
scaffold. We recommend that the user make their changes according to the structure and process described above. The user
 can then build a container from the context to test their changes before uploading to baseten.
