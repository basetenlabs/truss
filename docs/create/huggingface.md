# Create a Truss of a Hugging Face model

[Hugging Face](https://huggingface.co/) is a supported framework on Truss. To package a Hugging Face model, follow the steps below or run [this Google Colab notebook]().

### Install packages

If you're using a Jupyter notebook, add a line to install the `transformers` and `truss` packages. Otherwise, ensure the packages are installed in your Python environment.

```python
!pip install transformers truss
```

### Create an in-memory model

This is the part you want to replace with your own code. Using a Hugging Face transformer, build a machine learning model and keep it in-memory.

{% hint style="warning" %}
All Hugging Face models must be wrapped as a pipeline.
{% endhint %}

```python
from transformers import pipeline

model = pipeline('fill-mask', model='bert-base-uncased')
```

### Create a Truss

Use the `mk_truss` command to package your model into a Truss.

```python
from truss import mk_truss

tr = mk_truss(model, target_directory="huggingface_truss")
```

Check the target directory to see your new Truss!

### Serve the model

To get a prediction from the Truss, try running:

```python
tr.docker_predict({"inputs": "TODO"})
```

For more on running the Truss locally, see [local development](../develop/localhost.md).