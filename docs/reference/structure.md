# Truss directory structure

A Truss is a directory containing the packaged model. This reference details the files and folders in said directory and their contents.

```
config.yaml
examples.yaml
data/
    <data>
    <serialized model>
model/
    __init__.py
    model.py
```

### config.yaml

This file specifies the [configuration options](../develop/configuration.md) to be applied to the Truss.

### examples.yaml

This file provides [sample inputs](../develop/examples.md) for running your model.

### data/

This optional folder has the most varied contents, and enumerating everything that could go in here is beyond the scope of these docs. The most likely thing to find in here is a serialized model, but this folder can contain any dependencies for serving the model like data sets, weights, parameters, or any associated exports with a serialized model.

### model/

This folder contains the code that deserializes and runs the model, as well as the [pre- and post-processing functions](../develop/processing.md).

Here is a breakdown of the functions in `model/model.py`:

#### __init__

A model class is instantiated by a Truss model serving environment with the following parameters:
1. `config`: Provides access to the same config that's bundled with a truss, as a dictionary.
2. `data_dir`: Provides a pathlib directory where all the data bundled with the truss is provided.
3. `secrets`: This dictionary like object provides access to the secrets declared in the truss, but bound at runtime. The values returned by the secrets dictionary are dynamic, the secret value returned for the same key may be different over time, e.g. if it's updated. This means that when you update the secret values, and for many secrets it's a good practice to update them periodically, you don't have to redeploy the model.


This constructor can declare any subset of above parameters that it needs, they're bound by name as needed and the rest are omitted. One can omit all parameters or even omit the constructor.

#### load

The model class can declare a load method. This method is guaranteed to be
invoked before any prediction calls are made. This is a good place for
downloading any data needed by the model. One can do this in the constructor as
well, but it's not ideal to block the constructor for a long time as it might
affect initialization of other components. So load is where you'd want to do any
expensive i/o operations.

If omitted this method is considered to be no-op.

#### preprocess

This method allows preprocessing input to the model. Model input is passed to
this method and the output becomes input to the predict method below.

If omitted, this method is assumed to be identity.

#### predict

Perhaps the most critical method, this is the method called for making
predictions. This method of the model call is passed input and the returned
output is the model's prediction for that input.

#### postprocess

This method provides a way to modify the model output before returning. Output of the predict method is input to this method and the output of this method is what's returned to the caller.

If omitted, this method is assumed to be identity.
