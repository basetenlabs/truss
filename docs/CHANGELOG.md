# Changelog

Release notes for new versions of Truss, in reverse chronological order.

### Version 0.1.1

This release adds a new flag in a Truss' `config.yaml` file: `spec_version`.

**This flag is only necessary for using Truss with Baseten.** The meaning of the flag for Baseten users is explained below.

Until now, there were two rules limiting model interface on Baseten:

* All models must accept their input formatted as `{"inputs": []}`
* All models must deliver their output formatted as `{"predictions": []}`

These interface rules were originally created to make models more interoperable, but were too limiting, so we have removed them. However, to ensure backward compatibility, the new `spec_version` parameter is included in Truss. If the parameter is set to 1.0, the original interface is used, if 2.0, the new interface is used.

Under the 1.0 spec, calls to `baseten.predict` looked like:

```python
baseten.predict([])
```

With the 2.0 spec, calls to `baseten.predict` takes a dictionary equivalent of the request object expected by the model:

```python
baseten.predict({})
```

Other fixes:

* Fixes inference in iPython environments
* Prints Truss handle errors in notebooks
* Improved codespace developer experience

### Version 0.1.0 (initial release)

This release introduces Truss, and as such everything is new!
