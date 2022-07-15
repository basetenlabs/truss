# Pre- and post-processing

When you serialize a model, you want to keep the output as possible, containing just the model and its direct dependencies without custom logic or extra frameworks. This is great for keeping file sizes small and deserialization fast, but makes it tough to know what to do with the serialized model.

One feature that makes Truss different from model serialization frameworks is bundled pre- and post-processing. Most models intended for use in production systems will want to have these functions as part of the Truss.

## Pre-processing

ML models are picky eaters. One of the difficulties of using a model in a production system is that it may only take a certain type of input, specifically formatted. This means anyone and anything interacting with the model needs to be aware of the nuances of its input requirements.

Truss allows you to instead bundle pre-processing Python code with the model itself. This code is run on every call to the model before the model itself is run, giving you the chance to define your own input format.

Here is a pre-processing function that...

```python
Code sample
```

## Post-processing

Similarly, the output of a ML model can be messy or cryptic. Demystify the model results for your end user or format a web-friendly response in the post-processing function.

Here is a post-processing function that...

```python
Code sample
```