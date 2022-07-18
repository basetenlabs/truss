# Manually

Creating a Truss manually, from a serialized model, works with any model-building framework, including from-scratch bespoke models.

To get started, initialize the Truss with the following command:

```
truss init
```

### Truss structure

To build a Truss manually, you have to understand the package in much more detail than using it with a supported framework. Fortunately, that's why we wrote this doc.

To familiarize yourself with the structure of Truss, review the [structure reference](../reference/structure.md). A Truss only has three files that you need to interact with, and this tutorial is an opinionated guide to working through them.

### Adding the model binary

First, you'll need to add a model binary to your new Truss. On supported frameworks, this is provided automatically by the `mk_truss` command. For a custom Truss, it can come from many sources, such as:

* Pickling your model
* Serializing your model
* Downloading a serialized model from the internet

This file should be put in the folder `data/model/` as, for example, `model.joblib` (replace `joblib` with the appropriate extension for your serialized model).

This model binary must be de-serialized in the model class.

### Building the model

The model file implements the following functions, in order of execution:

* A constructor `__init__` to initiate the class
* A function called `load`, called once before any predictions
* A function `preprocess`, called once before each prediction
* A function `predict` that actually runs the model
* A function `postprocess`, called once after each prediction

Having both a constructor and a load function means you have flexibility on when you download and/or deserialize your model. Put these steps in the load function to run them on-demand on first prediction (though note that will cause a sort of "cold start" in your model service).

Also, your model gets access to certain values, including the `config.yaml` and `examples.yaml` file for configuration and sample input, respectively, as well as the `data` folder where you previously put the serialized model.

Once your model is created, you'll likely need to develop it further, see the next section for everything you need to know about local development!
