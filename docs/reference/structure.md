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

This file provides [sample inputs](../develop/sample-inputs.md) for running your model.

### model/

This folder contains the code that deserializes and runs the model, as well as the [pre- and post-processing functions](../develop/processing.md).

#### __init__.py

Always empty, generated only for Python packaging purposes.

#### model.py

Implements the class `Model` to serve the model.

### data/

This optional folder has the most varied contents, and enumerating everything that could go in here is beyond the scope of these docs. The most likely thing to find in here is a serialized model, but this folder can contain any dependencies for serving the model like data sets, weights, parameters, or any associated exports with a serialized model.