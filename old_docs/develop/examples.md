# Sample inputs

When you test your model locally, you likely have a go-to set of inputs to make sure everything is working. To improve the developer experience and document the input format, the `examples.yaml` file stores sample inputs. No more forgetting if your model takes a list or a dictionary, and no more copying inputs between model invocation tests.

Format each example as follows:

```yaml
- name: example_name
  input:
    inputs:
      - [10, 20, 30]
```

Now, in your terminal you can run:

```
truss run-example test_model --local
```

This will invoke your model locally and pass the examples as input.
