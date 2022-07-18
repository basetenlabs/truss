# Local model serving



The point of local development is to test pre- and post-processing. Deployment loop is very slow. Instead, use fast local iteration to make sure it works, and docker means that the environments matches production very closely.


## Staging checklist

Use Truss locally to validate that your model works on an environment that, thanks to Docker, closely resembles production.

Here are some things you might want to check:

* Dependencies: Are the Python packages you need available and working?
* System packages: Are system-level dependencies installed with the correct version?
* Environment variables: Do you have all of your config vars and API keys securely

Code, unfortunately, doesn't always work the first time. So local testing is essential to rapid iteration...

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

Testing through a docker image closely simulates the serving environment and is great for final testing. But it could be too slow for a tight dev loop. For a faster dev loop you can run prediction on the scaffold directory directly. Unlike docker image, this mechanism requires that you already have the right python requirements and system packages installed.

To test your model without building docker image do the following.

Install scikit-learn into your codespace
poetry add scikit-learn
Call scaffold predict locally scaffold predict test_model '{"inputs": [[10, 20, 30]]}' --local You should get the same result as before, but it should be a lot quicker and you won't see any docker image build logs. If you see an error saying `No module named 'sklearn' then you skipped step 1.
You can also specify examples for the model and run them instead. It's much easier to express request data in the example file. Running the example provides for a good dev loop.

Update the examples.yaml file under test_model with the following content:

example1:
  inputs:
    - [10, 20, 30]
Now run the example:

scaffold run-example test_model --local
You should see similar output as before. But this way, the dev loop is much easier, and having examples alongside the model will help others learn how to use it.
