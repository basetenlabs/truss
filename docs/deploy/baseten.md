# Deploy a Truss to Baseten

[Baseten](https://baseten.co), where Truss was originally developed, is a platform for building full-stack applications powered by ML models. You can deploy a model on Baseten with or without a Truss, but creating a Truss allows you to develop and test your model locally first.

To deploy a Truss on Baseten, you first need:

* A [Baseten account](https://app.baseten.co/accounts/signup/)
* An [API key](https://docs.baseten.co/settings/api-keys) on your account

Start by adding the Baseten Python client to your development environment.

```
pip install baseten
```

{% hint style="info" %}

If your model is already in memory (you created it with `mk_truss`), you can skip loading it into memory from the directory.

{% endhint %}

Before deploying your Truss, you may need to load it into memory in a Jupyter notebook or similar Python environment:

```python
import truss

my_truss = truss.from_directory("my_truss_lives_here")
```

Once your Truss is in memory, simply run the following:

```python
import baseten

baseten.login("PASTE_API_KEY_HERE")
baseten.deploy_truss(my_truss)
```

Head over to [your Baseten account](https://app.baseten.co) to see the model deployment logs and interface with your newly deployed model!
