# Local model serving

This is a guide for taking a Truss and running it locally.

{% hint style="info %}

Make sure you have [Docker installed](https://docs.docker.com/get-docker/) before serving a Truss locally

{% endhint %}

Start at the Truss' README file

Todo: Write about how to interface with the model locally

`python -m truss predict mytruss input`


If you want to load back the modified Truss into memory:

```
tr = truss.from_directory(mytruss)
```