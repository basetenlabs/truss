# Configuration

Truss is configurable to its core. Every Truss must include a file `config.yaml` in its root directory, which is automatically generated when the Truss is created. However, configuration is optional. Every configurable value has a sensible default, and a completely empty config file is valid.

The [iris sklearn Truss](../../examples/iris/config.yaml) has a good example of a typical Truss config:

```yaml
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

More examples can be found [here](https://github.com/basetenlabs/truss/tree/main/examples).

{% hint style="warning" %}

YAML syntax can be a bit non-obvious when dealing with empty lists and dictionaries. You may notice the following in the default Truss config file:

```yaml
requirements: []
secrets: {}
```

When you fill them in with values, list and dictionaries should look like this:

```yaml
requirements:
  - dep1 
  - dep2
secrets:
  - key1: default_value1
  - key2: default_value2
```

{% endhint %}

Let's investigate the various values that can be set in the config file.

### model_framework

There are many popular frameworks for creating ML models, and this field represents
the framework used. Many models are made with multiple frameworks, such as a keras model wrapping tensorflow inside. This field refers to the outermost layer of any such composite model. Supported values:

```yaml
custom
huggingface
keras
lightgbm
pytorch
sklearn
tensorflow
xgboost
```

This list is expected to grow. In fact, adding support for an ML framework
you're familiar with can be a good contribution to Truss.

### model_type

A model type is typically a subcategory under a model framework.

For example for HuggingFace the model_type field maps to the pipeline task such as `text-classification`.

### model_name

This is the display name of a model. This name may be used for identifying a model when deployed to model serving platforms such as Baseten.

### model_module_dir

A model in a Truss is ultimately a Python module. This is the directory in a truss that represents that module. This directory is called `model` by default.

### model_class_filename

Truss models are represented by a model class inside a python module. This field represents the file that holds that class. This file is called `model.py` by default.

### model_class_name

This is the class that represents a model in Truss. This class is called `Model` by default.

### data_dir

Most Truss models need access to a serialized model binary. For example, Tensorflow
models use the SavedModel representation, while Joblib and pickle are other common
serialization formats. Truss provides a way of bundling any blobs that are then
made available to the model class constructor. Any blobs simply need to be
placed in a data directory. This field is used to specify the name of that
directory under a truss and is called `data` by default.

When the model class is instantiated at serving time, the model class
constructor is passed a parameter called `data_dir`, which points to a pathlib
path to a directory on disk where the blobs are provided. This mechanism
provides flexibility to the model server on how to make the blobs available, and
ease to the model code.


For example, the [iris model](../../examples/iris/) bundles the model binary at
`data/model/model.joblib`.

The model constructor in the [model
class](../../examples/iris/model/model.py#L9) then uses the data_dir parameter
to load the model binary from.

### input_type

This field is reserved for representing the type of input that the model
expects. It isn't used currently, but is a good place to put this type
information for documentation.

### model_metadata

This is a catchall for any information that a model needs, that can't be
represented in one of the other fields. The whole config is available to the
model at runtime, so this is a good place to store any custom information that
model needs then. For example, sklearn models include a flag here that indicates whether
the model supports returning probabilities alongside predictions.

```yaml
model_metadata:
  supports_predict_proba: true
```

### requirements

This field represents the Python dependencies that the model module depends on.
The requirements should be provided in the [pip requirements file
format](https://pip.pypa.io/en/stable/reference/requirements-file-format/), but
as a list, with each list entry being the line in the requirements file.


```yaml
requirements:
- scikit-learn==1.0.2
- threadpoolctl==3.0.0
- joblib==1.1.0
- numpy==1.20.3
- scipy==1.7.3
```

### system_packages

Truss assumes the Debian operating system. This field can be used to specify any
system packages that you would typically install using `apt`.

```yaml
system_packages:
- ffmpeg
- libsm6
- libxext6
```

### environment_variables

{% hint style="danger" %}
Do not store secret values directly in environment variables (or anywhere in the config file). See the section on secrets below for information on properly managing secrets.
{% endhint %}

Any environment variables can be provided here as key value pairs and are
exposed to the environment that the model executes in. Many Python libraries can
be customized using environment variables, so this field can be quite handy in
those scenarios.

```yaml
environment_variables:
    AWS_ACCESS_KEY_ID: AKIAIOSFODNN7EXAMPLE
    AWS_SECRET_ACCESS_KEY: DummyDefaultToBeReplacedAtRuntime
```

### resources

Indicate runtime resources such as CPU, RAM and GPU that the
model ideally needs. These requirements are a suggestion, the model serving
environment may not always be able to honor them.

```yaml
resources:
  cpu: 800m
  memory: 1Gi
  use_gpu: true
```

### python_version

Version of python to use for serving the model. Currently supported python
version are:

1. py37 -- python 3.7
2. py38 -- python 3.8
3. py39 -- python 3.9

### examples_filename

Having inputs to try a model with is extremely handy, it makes the model much
more approachable. [Examples](examples.md) can be bundled with a Truss, in the form of a list.
Each element of the list should be a dictionary with:

1. name Name of the example
2. input Input for that example

Currently, this file should be yaml and is called `examples.yaml` by default.
This field can be used to customize the name of this file although there's
hardly ever a reason to.

The [`truss` cli](../reference/cli.md) has handy methods for listing and running examples.

### secrets

{% hint style="danger" %}

This field can be used to specify the keys for such secrets and dummy default
values --  ***Never store actual secret values in the config***. Dummy default
values are instructive of what the actual values look like and thus act as
documentation of the format.

{% endhint %}

A model may depend on certain secret values that can't be bundled with the model
and need to be bound securely at runtime. For example, a model may need to download
information from s3 and may need access to AWS credentials for that.

It's important that the model lists all the secret keys that it expects at runtime.
This not only serves as good practice but certain model serving frameworks may
make available only the secrets for which keys are listed here.

```yaml
secrets:
  openai_api_key:
```

Secret names need to confirm to the [k8s secret name guidelines](https://kubernetes.io/docs/concepts/configuration/secret/).

How secrets are mounted at runtime depends on the serving environment. For example, when
running on Docker locally, Truss allows specifying secrets in
`~/.truss/config.yaml`, while on Baseten, secrets can be specified as regular
[organization secrets](https://docs.baseten.co/settings/secrets).
