# Create a Truss from a serialized model

Command line: truss init, makes the structure of a Truss

* Start with structure of model file and fill it in
* Have to understand internals of Truss and Model
    * Model file with specific functions:
        * Load, predict, pre-process, post-process, constructor
    * Model gets access to certain things
        * Config, directory `data`, secrets
    * Access a model binary
        * Generated automatically by mk_truss
        * Like pickling a model, tensorflow save-model, protobuff, pytorch has a mechanism...
        * Serialized model
        * De-serialize it in model class
        * Download serialized model from internet
    * Function called load, called once, before any prediction
    * Constructor vs load means you have flexibility on when you download & deserialize, or on-demand on first prediction (like a cold start)
