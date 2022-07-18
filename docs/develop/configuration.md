# Configuration

Requires example values for every config field as well as entire example
configs.

Configuration is the anchor of a Truss. Every Truss has to have a file called
`config.yaml`, everything else is customizable, with good defaults.

As every field in the config has defaults, a completely empty `config.yaml` is
valid:
```config.yaml
```

The iris sklearn truss is a good example of a typical Truss config:

```config.yaml
model_framework: sklearn
model_metadata:
  model_binary_dir: model
  supports_predict_proba: true
python_version: py39
requirements:
- scikit-learn==1.0.2
- threadpoolctl==3.0.0
- joblib==1.1.0
- numpy==1.20.3
- scipy==1.7.3
```

More examples can be found [here](../../examples/).

## model_framework

There are many popular frameworks for creating ML models, this field represent
the framework used. Supported values are:

1. sklearn
2. tensorflow
3. keras
4. pytorch
5. huggingface
6. xgboost
7. lightgbm
8. custom

This list is expected to grow. In fact, adding support for an ML framework
you're familiar with can be a good contribution to Truss.

The framework types represent the outmost level the model is represented; a
model representated in one framework externally may be represented with another
one internally; a keras model may be tensorflow inside, a huggingface model may
be keras inside and a custom model may be wrapping practially any ML framework.

## model_type

A model type is typically a subcategory under a model framework.

For example for HuggingFace the model_type field maps to the pipeline task such
as `text-classification`.

## model_name

This is the display name of a model. e.g. this name may be used for identifying
a model when deployed to model serving platforms such as Baseten.

## model_module_dir

A model in a Truss is ultimately a python module. This is the directory in a
truss that represents that module. This directory is called `model` by default
and that's usually good enough.

## model_class_filename

Truss models are represented by a model class inside a python module. This field
represents the file that holds that class. This file is `model.py` by default
and that's usually good enough.

## model_class_name

This is the class that represents a model in Truss. By defaut this class is
`Model` and that's usually good enough.

### Model class protocol

#### __init__

A model class is instantiated by a Truss model serving environment with the following parameters:
1. config
This provides access to the same config that's bundled with a truss, as a dictionary.
2. data_dir
This provides a pathlib directory where all the data bundled with the truss is provided.
3. secrets This dictionary like object provides access to the secrets declared
in the truss, but bound at runtime. The values returned by the secrets
dictionary are dynamic, the secret value returned for the same key may be
different over time, e.g. if it's updated. This means that when you update the
secret values, and for many secrets it's a good practice to update them
periodically, you don't have to redeploy the model.


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
This method provides a way to modify the model output before returning. Output of the prefict method is input to this method and the output of this method is what's returned to the caller.

If omitted, this method is assumed to be identity.




## data_dir

Most Truss models need access to a serialized model binary. e.g. Tensorflow
models use the SavedModel representation. Joblib and pickle are other common
serialization formats. Truss provides a way of bundling any blobs that are then
made available to the model class constructor. Any blobs simply need to be
placed in a data directory. This field is used to specify the name of that
directory under a truss and is `data` by default.

When the model class is instantiated at serving time, the model class
constructor is passed a parameter called `data_dir`, which points to a pathlib
path to a directory on disk where the blobs are provided. This mechanism
provides flexibility to the model server on how to make the blobs available, and
ease to the model code.


e.g. the [iris model](../../examples/iris/) bundles the model binary at
`data/model/model.joblib`

The model constructor in the [model
class](../../examples/iris/model/model.py#L9) then uses the data_dir parameter
to load the model binary from.

## input_type

This field is reserved for representing the type of input that the model
expects. It isn't used currently, but is a good place to put this type
information for documentation.

## model_metadata

This is a catchall for any information that a model needs, that can't be
represented in one of the other fields. The whole config is available to the
model at runtime, so this is a good place to store any custom information that
model needs then. e.g. sklearn models include a flag here that indicates whether
the model supports returning probabilities alongside predictions.

```config.yaml
model_metadata:
  supports_predict_proba: true
```


## requirements

This field represents the python dependencies that the model module depends on.
The requirements should be provided in the [pip requirements file
format](https://pip.pypa.io/en/stable/reference/requirements-file-format/), but
as a list, each list entry being the line in the requirements file.

e.g.
```config.yaml
requirements:
- scikit-learn==1.0.2
- threadpoolctl==3.0.0
- joblib==1.1.0
- numpy==1.20.3
- scipy==1.7.3
```

## system_packages

Truss assumes the debian operating system. This field can be used to specify any
system packages that you would typically install using `apt`.

e.g.
```config.yaml
system_packages:
- ffmpeg
- libsm6
- libxext6
```

## environment_variables

Any environment variables can be provided here as key value pairs, these are
exposed to the environment that the model executes in. Many python libraries can
be customized using environment variables, this field can be quite handy in
those scenarios.

e.g.
```config.yaml
environment_variables:
    AWS_ACCESS_KEY_ID: AKIAIOSFODNN7EXAMPLE
    AWS_SECRET_ACCESS_KEY: DummyDefaultToBeReplacedAtRuntime
```

## resources

A model can indicate runtime resources such as cpu, memory and GPU that the
model ideally needs. These requirements are suggestive, model serving
environment may not always be able to honor them, but this information can be
quite valuable.

e.g.
```config.yaml
resources:
  cpu: 800m
  memory: 1Gi
  use_gpu: true
```

## python_version

Version of python to use for serving the model. Currently supported python
version are:
1. py37 -- python 3.7
2. py38 -- python 3.8
3. py39 -- python 3.9

Clearly, more will be added in future.

## examples_filename

Having inputs to try a model with is extremely handy, it makes the model much
more approachable. Examples can be bundled with a Truss, in the form of a list.
Each element of the list should be a dictionary with:
1. name Name of the example
2. input Input for that example

Currently, this file should be yaml and is called `examples.yaml` by default.
This field can be used to customize the name of this file although there's
hardly ever a reason to.

The `truss` cli has handy methods for listing and running examples. todo: Add
link

## secrets
A model may depend on certain secret values that can't be bundled with the model
and need to be bound securely at runtime. e.g. A model may need to download
information from s3 and may need access to AWS credentials for that.

This field can be used to specify the keys for such secrets and dummy default
values --  ***Never store actual secret values in the config***. Dummy default
values are instructive of what the actual values look like and thus act as
documentation of the format.

It's important that the model lists all the secret keys that it expects at runtime.
This not only serves as good practice but certain model serving frameworks may
make available only the secrets for which keys are listed here.

e.g.
```config.yaml
secrets:
  openai_api_key:
```

Secret names need to confirm to the k8s secret name guidelines.

How secrets are mounted at runtime depends on the serving environment. e.g. when
running on docker locally, Truss allows specifying secrets in
`~/.truss/config.yaml`, on Baseten, secrets can be specified as regular
organization secrets.
