# Create a Truss of a Hugging Face model

[Hugging Face](https://huggingface.co/) is a supported framework on Truss. To package a Hugging Face model, follow the steps below or run [this colab notebook]().

### Install packages

If you're using a Jupyter notebook, add a line to install the `transformers` and `truss` packages. Otherwise, ensure the packages are installed in your Python environment.

```python
!pip install transformers truss
```

### Create an in-memory model

This is the part you want to replace with your own code. Using a Hugging Face transformer, build a machine learning model and keep it in-memory.

```python
from transformers import pipeline

model = pipeline('fill-mask', model='bert-base-uncased')
```

### Create a Truss

Use the `mk_truss` command to package your model into a Truss.

```python
from truss import mk_truss

mk_truss(model, target_directory="huggingface_truss")
```

Check the target directory to see your new Truss!