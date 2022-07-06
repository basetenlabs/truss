# Model class

The model class is the entry point to your model and is the most important part
of a Truss. Model serving system looks for and invokes certains
methods on the model class to make predictions.

## model initialization

When the Model class is instantiated it's passed certain information as kwargs

1. `data_dir` This pathlib.Path provides access to the data bundled with a
   Truss. This path should be considered read-only by the model.

2.  `config` The config argument provides access to the full Truss configuration
to the model.


## load

This method is invoked before any prediction requests are made, to provide a
chance to prepare. This is the place where a model may want to do expensive work
such as downloading data, loading model binaries into memory etc.

## predict

This is the essential prediction endpoint. The model is provides input here in
the form of a dictionary and the output is the prediction. It's customary for
the request to have a field called `inputs` or `instances` where multiple inputs
can be provided to get predictions in a batch. A single prediction then, becomes
just a batch of one.

```
{
    'inputs': [...],
}
```

## preprocess

This method is called before the predict method and provides a chance to do any
pre-processing. Output of this method is input to the predict method. Default is
to pass through the input as is. Nothing stops one from doing any pre-processing
in the predict method itself, the preprecess method is just a code organization
suggestion.

## postprocess

This method is called after the predict method. Output of predict method is
provided as input to this method and provides a chance to post-process. Default
is to passthrough.
