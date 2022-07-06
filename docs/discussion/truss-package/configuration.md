# Configuration

Configuration is provided in the Truss package in a file called config.yaml.
The name of this file is fixed, everything else in the package is configurable
through this config file though.

The config is defined at
`truss.truss/truss_config.py`

Some of the most important configurations follow.

## Model serving environment set up

### Python requirements
Any python requirements that are needed by the model module can be specified
here in the standarad pip requirements.txt format (todo: Add link).

### System packages
Truss generates a debian image with some basic system packages. Any
additional system packages needed by the model module can be specified here.

### Environment variables
Any environment variables that the model code depends on can be specified here.

### Resources
A running model needs compute resources such as cpu, memory and GPU. These can
be specified here. If the model serving environment is unable to fulfill these
system requirements then it may deny the deployment or pick reduced resources,
the decision is left to the model serving environment. e.g. If a GPU is not
available then the model serving environment may run the model on CPU. The
settings here indicate the preferred resource requirements of the model.

### Model metadata
Model framework can be sklearn, pytorch, keras, tensorflow, huggingface or
custom. This knowledge may allow the serving environment to include standard
libraries and packages for these frameworks.

Any other information about the model pay be specified in the model_metadata
dictionary.

For some model frameworks, model_type provides further information of the type
of model being executed.

Model name is mostly for display purposes, e.g. this may be the name the Model
is listed as on the models page in the model serving system's web interface.

## Package structure configuration
Except for the name of the configuration file, which should be config.yaml,
names of other parts of Truss configuration can be modified. e.g. by default,
Truss will look for a class named `Model` under `model/model.py` for the
model interaction code. But, if you wish, you can customize it to look for
`SuperModel` under `my_preferred/model/location/super_model.py`.
