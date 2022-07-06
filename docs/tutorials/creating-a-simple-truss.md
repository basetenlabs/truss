# Creating a Simple Truss

As a simple example we will look at the creation of a scaffold using an Scikit-learn classifier. We start with a trained model described below.

```python
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

iris = load_iris()
data_x = iris['data']
data_y = iris['target']
rfc_model = RandomForestClassifier()
rfc_model.fit(data_x, data_y)
```

And now we can create a scaffold from the in-memory model

```python
from baseten_scaffolding.scaffold.build import scaffold
scaffold = scaffold(rfc_model, target_directory='test_rfc')
```

This will produce a folder `test_rfc/` relative to the current directory which will contain all the elements required to build a scaffold container. In fact, it can produce the command to build the container.

```python
>> scaffold.docker_build_string
'docker build  -f test_rfc/sklearn-server.Dockerfile test_rfc'
```

It is wise to append a target via `-t <name>` to the build. You can build and run the container like so

```
docker build  -f test_rfc/sklearn-server.Dockerfile test_rfc -t test_rfc
docker run --rm  -p 8080:8080 -t test_rfc
```

And then curl a POST to the server on your localhost

```
curl -H 'Content-Type: application/json' -d '{"inputs": [[0,0,0,0]]}' -X POST http://localhost:8080/v1/models/model:predict
```

Congrats! You just built a scaffold and tested it locally.
