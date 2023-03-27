# Create a Truss of a Hugging Face model

[Hugging Face](https://huggingface.co/) is a supported framework on Truss. To package a Hugging Face model, follow the steps below or run [this Google Colab notebook](https://colab.research.google.com/github/basetenlabs/truss/blob/main/docs/notebooks/huggingface_example.ipynb).

### Install packages

If you're using a Jupyter notebook, add a line to install the `transformers` and `truss` packages. Otherwise, ensure the packages are installed in your Python environment.

```python
!pip install --upgrade transformers truss
```

{% hint style="warning" %}
Truss officially supports `transformers` version 4.21.0 or higher. Especially if you're using an online notebook environment like Google Colab or a bundle of packages like Anaconda, ensure that the version you are using is supported. If it's not, use the `--upgrade` flag and pip will install the most recent version.
{% endhint %}

### Create an in-memory model

This is the part you want to replace with your own code. Using a Hugging Face transformer, build a machine learning model and keep it in-memory. In this example we're using [bert-base-uncased](https://huggingface.co/bert-base-uncased), which will fill in the missing word in a sentence.

{% hint style="danger" %}
All Hugging Face models must be wrapped as a pipeline.
{% endhint %}

```python
from transformers import pipeline

model = pipeline('fill-mask', model='bert-base-uncased')
```

### Create a Truss

Use the `create` command to package your model into a Truss.

```python
from truss import create

tr = create(model, target_directory="huggingface_truss")
```

Check the target directory to see your new Truss!

### Serve the model

To get a prediction from the Truss, try running:

```python
tr.predict("Donatello is a teenage mutant [MASK] turtle")
```

For more on running the Truss locally, see [local development](../develop/localhost.md).
